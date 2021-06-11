#include <iostream>
#include <vector>
#include "tkDNN/tkdnn.h"
#include "tkDNN/test.h"
#include "tkDNN/DarknetParser.h"

int main() {
    std::string bin_path  = "yolo3_coco4";
    std::vector<std::string> input_bins = { 
        bin_path + "/layers/input.bin"
    };
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer82_out.bin",
        bin_path + "/debug/layer94_out.bin",
        bin_path + "/debug/layer106_out.bin"
    };
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = std::string(TKDNN_PATH) + "/tests/darknet/cfg/yolo3_coco4.cfg";
    std::string name_path = std::string(TKDNN_PATH) + "/tests/darknet/names/coco4.names";
    downloadWeightsifDoNotExist(input_bins[0], bin_path, "https://cloud.hipert.unimore.it/s/o27NDzSAartbyc4/download");

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    //convert network to tensorRT
    tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName(bin_path.c_str()));
    
    int ret = testInference(input_bins, output_bins, net, netRT);
    net->releaseLayers();
    delete net;
    delete netRT;
    return ret;
}
