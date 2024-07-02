/*
 * Author: alm
 * Email: almtach@outlook.com
 */

#include <ros/ros.h>
#include <ros/package.h>
#include "object_information_msgs/Object.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "postprocess.h"
#include "rk_common.h"
#include "rknn_api.h"
#include "config.h" 
// COCO 数据集标签
static const char *labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};
static const unsigned char colors[19][3] = {
    {54, 67, 244},
    {99, 30, 233},
    {176, 39, 156},
    {183, 58, 103},
    {181, 81, 63},
    {243, 150, 33},
    {244, 169, 3},
    {212, 188, 0},
    {136, 150, 0},
    {80, 175, 76},
    {74, 195, 139},
    {57, 220, 205},
    {59, 235, 255},
    {7, 193, 255},
    {0, 152, 255},
    {34, 87, 255},
    {72, 85, 121},
    {158, 158, 158},
    {139, 125, 96}};
// RKNN 应用上下文
rknn_app_context_t rknn_app_ctx;

// 阈值变量
float box_conf_threshold;
float nms_threshold;
ros::Time last_time;
object_information_msgs::Object objMsg;
ros::Publisher obj_pub;
sensor_msgs::ImagePtr image_msg;
image_transport::Publisher image_pub;
// detect_result_group_t detect_result_group;
bool display_output = true;
std::string camera_topic;
std::string label_flag = "none";
// 图像订阅回调函数
void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // 将 ROS 图像消息转换为 OpenCV 图像
    ros::Time current_time = ros::Time::now();
    int color_index = 0;
    cv_bridge::CvImagePtr cv_ptr;
    float fps = 0.0;
    
    // last_time = current_time;
    
    try
    {

        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat orig_img = cv_ptr->image;
    cv::Mat img;
    cv::Mat resized_img;

    // 转换 BGR -> RGB
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;

    // 检查尺寸
    int width = rknn_app_ctx.model_width;
    int height = rknn_app_ctx.model_height;
    void *buf = nullptr;
    if (img_width != width || img_height != height)
    {
        cv::resize(img, resized_img, cv::Size(width, height));
        buf = (void *)resized_img.data;
    }
    else
    {
        buf = (void *)img.data;
    }

    // 声明检测结果列表
    object_detect_result_list od_results;

    // 设置输入数据
    rknn_input inputs[rknn_app_ctx.io_num.n_input];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
    inputs[0].buf = buf;

    // 设置输入
    int ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
    if (ret < 0)
    {
        ROS_ERROR("rknn_input_set fail! ret=%d", ret);
        return;
    }

    // 分配输出
    rknn_output outputs[rknn_app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < rknn_app_ctx.io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!rknn_app_ctx.is_quant);
    }

    // 运行模型
    rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
    rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);

    // 后处理
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;
    post_process(&rknn_app_ctx, outputs, box_conf_threshold, nms_threshold, scale_w, scale_h, &od_results);

    // 绘制对象
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        const unsigned char *color = colors[color_index % 19];
        color_index++;
        cv::Scalar cc(color[0], color[1], color[2]);
        object_detect_result *det_result = &(od_results.results[i]);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cc);
        objMsg.header.seq++;
        objMsg.header.stamp = current_time;
        objMsg.probability = det_result->prop;
        objMsg.label = labels[det_result->cls_id];
        objMsg.position.position.x = x1;
        objMsg.position.position.y = y1;
        objMsg.position.orientation.x = x1 + int((x2 - x1) / 2);
        objMsg.position.orientation.y = y1 + int(((y2 - y1) / 2));
        objMsg.size.x = y2 - y1;
        objMsg.size.y = x2 - x1;
        obj_pub.publish(objMsg);
        // 添加标签
        // int length = 10;
        // // 绘制水平线
        // cv::line(orig_img, cv::Point(x1 + int((x2 - x1) / 2) - length, y1 + int(((y2 - y1) / 2))), cv::Point(x1 + int((x2 - x1) / 2) + length, y1 + int(((y2 - y1) / 2))), (0, 0, 255), 2);
        // // 绘制垂直线
        // cv::line(orig_img, cv::Point(x1 + int((x2 - x1) / 2), y1 + int(((y2 - y1) / 2)) - length), cv::Point(x1 + int((x2 - x1) / 2), y1 + int(((y2 - y1) / 2)) + length), (0, 0, 255), 2);
        // cv::circle(orig_img,cv::Point(x1+int((x2-x1)/2),y1+int(((y2-y1)/2))),10,(0,0,255));
        sprintf(text, "%s %.1f%%", labels[det_result->cls_id], det_result->prop * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = det_result->box.left;
        int y = det_result->box.top - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > orig_img.cols)
            x = orig_img.cols - label_size.width;

        cv::rectangle(orig_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::putText(orig_img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
    if (display_output)
    {
        //   cv::Mat out_image ;
        char text[32];
        
        // last_time = current_time;
        fps = 1.0 / (ros::Time::now() - current_time).toSec();
        sprintf(text, "RKNN YOLOV8 FPS %d", int(fps));
        cv::putText(orig_img, text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", orig_img).toImageMsg();
        image_pub.publish(image_msg);
    }

    last_time = current_time;

    // 释放输出
    ret = rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

    // 显示结果
    // cv::imshow("Object Detection", orig_img);
    // cv::waitKey(30);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_detection_node");
    ros::NodeHandle nh("~");
    ros::NodeHandle n;
    // 初始化 RKNN 上下文
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // 加载模型
    // std::string node_name = ros::this_node::getName();
    // const std::string package_name = "rknn_yolov8_ros";
    const std::string package_name = PROJECT_NAME; 
    std::string chip_type;
    nh.param("chip_type", chip_type, std::string("rk3588"));
    std::string path = ros::package::getPath(package_name) + ("/models/") + chip_type + ("/");
    std::string model_name;

    // std::string path = ros::package::getPath(package_name)+("/models/")+chip_type+("/");
    nh.param<std::string>("model_name", model_name, "yolov8s.rknn");
    int model_data_size = 0;
    unsigned char *model_data = load_model((path + model_name).c_str(), model_data_size);

    int ret = rknn_init(&rknn_app_ctx.rknn_ctx, model_data, model_data_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        ROS_ERROR("rknn_init fail! ret=%d", ret);
        return -1;
    }

    // 查询 SDK 版本
    rknn_sdk_version version;
    ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        ROS_ERROR("rknn_init error ret=%d", ret);
        return -1;
    }
    ROS_INFO("sdk version: %s driver version: %s", version.api_version, version.drv_version);

    // 获取模型输入输出数量
    ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &rknn_app_ctx.io_num, sizeof(rknn_app_ctx.io_num));
    if (ret != RKNN_SUCC)
    {
        ROS_ERROR("rknn_query fail! ret=%d", ret);
        return -1;
    }
    ROS_INFO("model input num: %d", rknn_app_ctx.io_num.n_input);

    // 获取输入属性
    rknn_tensor_attr input_attrs[rknn_app_ctx.io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < rknn_app_ctx.io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            ROS_ERROR("rknn_init error ret=%d", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 获取输出属性
    rknn_tensor_attr output_attrs[rknn_app_ctx.io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < rknn_app_ctx.io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 设置上下文
    rknn_app_ctx.input_attrs = (rknn_tensor_attr *)malloc(rknn_app_ctx.io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, rknn_app_ctx.io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr *)malloc(rknn_app_ctx.io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, rknn_app_ctx.io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        ROS_INFO("model input is NCHW");
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height = input_attrs[0].dims[2];
        rknn_app_ctx.model_width = input_attrs[0].dims[3];
    }
    else
    {
        ROS_INFO("model input is NHWC");
        rknn_app_ctx.model_height = input_attrs[0].dims[1];
        rknn_app_ctx.model_width = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    ROS_INFO("model input height=%d, width=%d, channel=%d",
             rknn_app_ctx.model_height, rknn_app_ctx.model_width, rknn_app_ctx.model_channel);

    // 设置阈值

    nh.param<std::string>("label_flag", label_flag, "cup");
    nh.param<float>("box_conf_threshold", box_conf_threshold, 0.5);
    nh.param<float>("nms_threshold", nms_threshold, 0.45);
    nh.param<std::string>("camera_topic", camera_topic, "/usb_cam/image_raw");
    // 订阅图像话题
    // ROS node relative

    image_transport::ImageTransport it(n);
    image_pub = it.advertise("/rknn_yolov8_image", 1);
    obj_pub = nh.advertise<object_information_msgs::Object>("/objects", 50);
    ros::Subscriber sub = nh.subscribe(camera_topic, 1, imageCallback);

    ros::spin();

    // 清理
    if (rknn_app_ctx.input_attrs != NULL)
        free(rknn_app_ctx.input_attrs);
    if (rknn_app_ctx.output_attrs != NULL)
        free(rknn_app_ctx.output_attrs);
    if (rknn_app_ctx.rknn_ctx != 0)
        rknn_destroy(rknn_app_ctx.rknn_ctx);

    return 0;
}
