# Frigate Modifications by Monica 

This documentation explains the changes I have made in the Frigate codebase.

Features added to the existing code base:

Attribute detection models:

* Human attribute detection model
* Vehicle attribute detection model

Human action detection model:

* Human falling action detection model
* Human fighting action detection model
* Human calling action detection model

All these models are integrated into classification module of the frigate. The config file of the frigate has to be altered to use these models. The results of the detection models will be displayed in the explore tab of the UI along with the detection. These classification models are secondary models i.e the primary object detection happens through any conventional object detection model such as yolo, and second level detection describing the detected person's attributes or actions, or detected car's attributes can be performed only on the detected respective objects. To enable this classification models, it is important to enable semantic search. The detection modules need to be in .onnx format and placed in the folder whose path is given in the config file. A typical config file changes are given below:

********Config file modification for the changes ***************

*
*
Config file like conventional frigate 
*
*
*
*
semantic_search:
  enabled: true
classification:
  human_attr:
    enabled: true
    threshold: 0.6
    human_attr_model_path:  /models/human_attr/human_attr.onnx
    human_attr_label_path: /models/human_attr/attr_labels.txt
  # vehicle_attr:
  #   enabled: true
  #   threshold: 0.6
  #   vehicle_attr_model_path:  /models/vehicle_attribute_model/model.onnx
  #   vehicle_attr_label_path: /models/vehicle_attribute_model/vehicle_attr_label.txt
  # fall_det:
  #   enabled: true
  #   threshold: 0.6
  #   falling_model_path:  /models/human_falling/falling_detection.onnx

  # call_det:
  #   enabled: true
  #   threshold: 0.6
  #   calling_model_path:  /models/human_calling/calling_detection.onnx
    


********End of Config file modification for the changes ***************


* Human attribute detection: 

Paddle attribute detection model has been integrated into the code base. Pedestrian attribute recognition has been widely used in the intelligent community, industrial, and transportation monitoring. Many attribute recognition modules have been gathered in PP-Human, including gender, age, hats, eyes, clothing and up to 26 attributes in total. Also, the pre-trained models are offered here and users can download and use them directly.

  Model download location:
| High-Precision Model    |  PP-HGNet_small  |  mA: 95.4  | per person 1.54ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.tar) |
| Fast Model    |  PP-LCNet_x1_0  |  mA: 94.5  | per person 0.54ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.tar) |
| Balanced Model    |  PP-HGNet_tiny  |  mA: 95.2  | per person 1.14ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_person_attribute_952_infer.tar) |

Instruction:
 * place the attribute detection model in .onnx format in a folder
 * place the models label files in .txt format in the folder
 * Human attribute detection if enabled runs on the detected "person" snippets
 * the detected attributes are displayed on the frigate webUI
 * model input size is (192, 256)

for details regarding attribute detection model kindly refer to paddledetection or pphuman documentation

* Vehicle attribute detection model

Vehicle attribute recognition is widely used in smart cities, smart transportation and other scenarios. In PP-Vehicle, a vehicle attribute recognition module is integrated, which can identify vehicle color and model.

| Task | Algorithm | Precision | Inference Speed | Download |
|-----------|------|-----------|----------|---------------------|
| Vehicle Detection/Tracking | PP-YOLOE | mAP 63.9 | 38.67ms | [Inference and Deployment Model](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Vehicle Attribute Recognition | PPLCNet | 90.81 | 7.31 ms | [Inference and Deployment Model](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip) |

- The provided pre-trained model supports 10 colors and 9 models, which is the same with VeRi dataset. The details are as follows:

# Vehicle Colors
- "yellow"
- "orange"
- "green"
- "gray"
- "red"
- "blue"
- "white"
- "golden"
- "brown"
- "black"

# Vehicle Models
- "sedan"
- "suv"
- "van"
- "hatchback"
- "mpv"
- "pickup"
- "bus"
- "truck"
- "estate"

Instruction:
 * place the attribute detection model in .onnx format in a folder
 * place the models label files in .txt format in the folder
 * vehicle attribute detection if enabled runs on the detected "vehicle" snippets. all the classes that comes in the category of vehicle from coco dataset is used
 * the detected attributes are displayed on the frigate webUI
 * model input size is (256, 192)


* Action detection model
  There are 3 action detection models integrated from frigate. Namely
    * Calling detection model
    * falling detection model
    * fighting detection model

  Download location:

 * Calling detection model:
   This model detects calling action in the video. 

    | Calling Recognition | PP-HGNet | Precision Rate: 86.85 | Single Person 2.94ms | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.pdparams) | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) |

 * Falling action detection model:
    This model detects when a person falls. It is to be noted this model works on a series of 8 frames. i.e all 8 frames are used to detect if there is falling action happening in the scene. If the falling happens the label is displayed in the 8th frame. There is a possibility that the image corresponding to the 8th frame is not having any falling scene as such. 

    | Falling Recognition            | ST-GCN    | Precision Rate: 96.43     | Single Person 2.7ms                 | - |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip)                      |

 * Fighting detection model:
    This model detects when there is fight in the scene. It is to be noted this model works on a series of 8 frames. i.e all 8 frames are used to detect if there is fighting action happening in the scene. If the fighting happens the labelis displayed in the 8th frame. There is a possibility that the image corresponding to the 8th frame is not having any fighting scene as such. 

    | Fighting Recognition | PP-TSM | Precision Rate: 89.06% | 128ms for a 2sec video | [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

