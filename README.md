# cards-training

1) Setup Variables in Gcloud VM
```bash
    export PROJECT_ID=<YOUR-PROJECT-ID>
    export ZONE=us-central1-a
    export INSTANCE_NAME=<VM-INSTANCE-NAME>
```
E.g. Mike's details
```bash
    export PROJECT_ID=project-2019-3mega-01
    export ZONE=us-central1-a
    export INSTANCE_NAME=vm-obj-det-tpu-ubuntu
```
E.g. 
```bash
    export PROJECT_ID=tpu-training-2
    export ZONE=us-central1-a
    export INSTANCE_NAME=obj-det-30min-tpu-ubuntu-2019-11-12-01-5028-4460
```

2) Start VM
```bash
    gcloud compute instances start $INSTANCE_NAME --zone $ZONE
```

3) SSH into VM
```bash
    gcloud compute ssh --project $PROJECT_ID --zone $ZONE $INSTANCE_NAME
```

4) List docker images
```bash
    docker image list
```

5) Set Docker Image Variable
```bash
     export DOCKER_IMG_NAME=detect-tf-image
```

6) Run Docker Image
```bash
    sudo docker run --rm -it --privileged -p 6006:6006 $DOCKER_IMG_NAME
    # Re-initialize the variables from Step 1 in Docker Container
    # These variables will be needed in Step 22
```

7) Change directory to models
```bash
     cd models
```

8) Git clone cards-training
```bash 
   git clone https://github.com/mmmwembe/cards-training.git
```

9) Generate CSV files from Images/Annotations
```bash
    cd cards-training
    python xml_to_csv.py 

    mkdir tfrecords
    cd ..
```
10) Extend Python Path
```bash
    echo $PYTHONPATH
    export PYTHONPATH=${PYTHONPATH}:${PWD}/:${PWD}/models
    export PYTHONPATH=${PYTHONPATH}:${PWD}/:${PWD}/cards-training
    echo $PYTHONPATH
```

11) Generate TFRecords
```bash
    cd cards-training

    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=tfrecords/train.record
    
    python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=tfrecords/test.record
```

12) Check location and size of tfrecords
```bash
   ll -h tfrecords/
```

13) Initialize Gcloud
```bash
    gcloud init
```

14) Setup BUCKET variables
# Create the following directory structure manually from Gcloud
# BUCKET ---
#     ----data
#     ----train
#     ----tflite
#     ----android-app
#     ----graph
```bash
    export BUCKET=<YOUR_BUCKET_NAME>
```
example - Michael
```bash
    export BUCKET=tpu-cards-training
```
example - Mambwe
```bash
    export BUCKET=mm-tpu-cards-training
```
# Make sure the pipeline.config (Michael) and pipeline0.config (Mambwe) are configured for YOUR_BUCKET
# Go to cards-training/pipeline
# wget "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config"

15) Copy TFrecords and labelmap.pbtxt to BUCKET
```bash
    gsutil cp tfrecords/train.record gs://${BUCKET}/data/
    gsutil cp tfrecords/test.record  gs://${BUCKET}/data/
    gsutil cp labels/labelmap.pbtxt  gs://${BUCKET}/data/

    gsutil cp pipeline/pipeline.config gs://${BUCKET}/data/ # Michael use pipeline.config; 
    gsutil cp pipeline/pipeline0.config gs://${BUCKET}/data/ #Mambwe use pipeline0.config
```

16) Check that TFrecords and labelmap.pbtxt are in BUCKET data directory
```bash
    gsutil ls gs://${BUCKET}/data/
```

17) change directory to tmp
```bash
    cd /tmp
    rm -r *
```

18) Download SSD MobileNet model and tar it
```bash
    curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz

    tar xzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
```

19) Copy the tarred ANN model to BUCKET/data
```bash
    gsutil cp /tmp/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt.* gs://${BUCKET}/data/
```

20) Setup TPU service account
```bash
    curl -H "Authorization: Bearer $(gcloud auth print-access-token)"  \
        https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig
```

21) Assign TPU service account
```bash
    export TPU_SVC_ACCOUNT=<YOUR_TPU_SVC_ACCOUNT_NAME>

    export TPU_SVC_ACCOUNT=service-402993653798@cloud-tpu.iam.gserviceaccount.com
```

22) Grant the ml.serviceAgent role to your TPU service account
```bash
    gcloud projects add-iam-policy-binding $PROJECT_ID  \
    --member serviceAccount:$TPU_SVC_ACCOUNT --role roles/ml.serviceAgent
```
23) Change directory to cd tensorflow/models/research/
```bash
    cd /tensorflow/models/research
```

24) Package the Tensorflow Object Detection code, run the following commands from the tensorflow/models/research/ directory:
# From models/research
```bash
    bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
    python setup.py sdist
    (cd slim && python setup.py sdist)
```

25) Run TPU Training
# Run TPU training - following examples below
# Michael
```bash
    gcloud ai-platform jobs submit training three_mega_cards_object_detection_`date +%s` \
    --job-dir=gs://${BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_tpu_main \
    --runtime-version 1.14 \
    --scale-tier BASIC_TPU \
    --region us-central1 \
    -- \
    --model_dir=gs://${BUCKET}/train \
    --tpu_zone us-central1 \
    --pipeline_config_path=gs://${BUCKET}/data/pipeline.config 
```
# Mambwe
```bash
    gcloud ai-platform jobs submit training three_mega_cards_object_detection_`date +%s` \
    --job-dir=gs://${BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_tpu_main \
    --runtime-version 1.14 \
    --scale-tier BASIC_TPU \
    --region us-central1 \
    -- \
    --model_dir=gs://${BUCKET}/train \
    --tpu_zone us-central1 \
    --pipeline_config_path=gs://${BUCKET}/data/pipeline0.config 
```
26) Run Evaluation (Right After Running Training)
# Michael
```bash
 gcloud ai-platform jobs submit training three_mega_cards_object_detection_eval_validation_`date +%s` \
    --job-dir=gs://${BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --runtime-version 1.14 \
    --scale-tier BASIC_GPU \
    --region us-central1 \
    -- \
    --model_dir=gs://${BUCKET}/train \
    --pipeline_config_path=gs://${BUCKET}/data/pipeline.config \
    --checkpoint_dir=gs://${BUCKET}/train
```
# Mambwe
```bash
 gcloud ai-platform jobs submit training three_mega_cards_object_detection_eval_validation_`date +%s` \
    --job-dir=gs://${BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --runtime-version 1.14 \
    --scale-tier BASIC_GPU \
    --region us-central1 \
    -- \
    --model_dir=gs://${BUCKET}/train \
    --pipeline_config_path=gs://${BUCKET}/data/pipeline0.config \
    --checkpoint_dir=gs://${BUCKET}/train
```

27) Check status of the training
```bash
     export JOB_NAME=<YOUR_JOB_NAME>
     gcloud ai-platform jobs describe $JOB_NAME
```

29) In another VM, run tensorboard
# Run Tensorboard in the VM host and not the docker container
```bash
    gcloud init
    export BUCKET=<YOUR_BUCKET_NAME>
    tensorboard --logdir=gs://${BUCKET}/train --host=localhost --port=8080
```
# Example - Michael
```bash
    gcloud init
    export BUCKET=tpu-cards-training
    tensorboard --logdir=gs://${BUCKET}/train --host=localhost --port=8080
```

# Example - Mambwe
```bash
    gcloud init
    export BUCKET=mm-tpu-cards-training
    tensorboard --logdir=gs://${BUCKET}/train --host=localhost --port=8080
```
30) Set up Transfer Variables
# Michael
```bash
    export CONFIG_FILE=gs://${BUCKET}/data/pipeline.config
    export CHECKPOINT_PATH=gs://${BUCKET}/train/model.ckpt-2000
    export OUTPUT_DIR=/tmp/tflite
```
# Mambwe
```bash
    export CONFIG_FILE=gs://${BUCKET}/data/pipeline0.config
    export CHECKPOINT_PATH=gs://${BUCKET}/train/model.ckpt-2000
    export OUTPUT_DIR=/tmp/tflite
```  
31) Obtain Graph for tflite

```bash
    python object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path=$CONFIG_FILE \
    --trained_checkpoint_prefix=$CHECKPOINT_PATH \
    --output_directory=$OUTPUT_DIR \
    --add_postprocessing_op=true
```

32) Transfer tflite_graph to gs://BUCKET/graph
```bash
    gsutil cp -r /tmp/tflite/* gs://${BUCKET}/tflite_graph/
```

33) Change directory to tensorflow
```bash
    cd ../..
```

34) Generate optimized TFLite model

```bash
 time bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=$OUTPUT_DIR/tflite_graph.pb \
    --output_file=$OUTPUT_DIR/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
```

35) Copy TFLite models (tflite_graph.pbtxt and tflite_graph.pb, detect.tflite) to gs://${BUCKET}/tflite/
```bash
     gsutil cp $OUTPUT_DIR/* gs://${BUCKET}/tflite/
```

36) QA Test for TFlite Model Prior to Deployment
## References:
[testing tflite model](https://medium.com/towards-artificial-intelligence/testing-tensorflow-lite-image-classification-model-e9c0100d8de3)

[part 1 - testing tflite](https://thinkmobile.dev/testing-tensorflow-lite-image-classification-model/)

[part 2 - testing tflite](https://medium.com/towards-artificial-intelligence/automate-testing-of-tensorflow-lite-model-implementation-5d1e919f7e07)

(a) Create qa_test model directory
```bash
    mkdir "qa_test"
```
(b) copy the TFlite model from Gcloud to qa_test
```bash
    gsutil cp gs://${BUCKET}/tflite/detect.tflite qa_test
```
(c) Set Model Variable
```bash
    TFLITE_QUANT_MODEL="qa_test/detect.tflite"
```
(d) Initialize Tflite model
```bash
    tflite_interpreter = tf.lite.Interpreter(model_path=$TFLITE_QUANT_MODEL)
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])
```

# >> == Input details ==
# >> shape:  [1, 300, 300, 3]
# >> type: <class 'uint8'>
# >> == Output details ==
# >> shape: [1 4]
# >> type: <class 'numpy.float32'>

# https://www.tensorflow.org/lite/convert/python_api#api

A) in VM Install tensorflow lite interpreter-only package
# https://www.tensorflow.org/lite/guide/python
```bash
    pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl
```
B) Gcloud init
```bash
    gcloud init
```
C) Set BUCKET variable
```bash

    export BUCKET=tpu-cards-training    # Michael

    export BUCKET=mm-tpu-cards-training # Mambwe
```
D) Download detect.tflite to /tmp

```bash
    gsutil cp gs://$BUCKET/tflite/detect.tflite /tmp/
```
E) Git Clone cards-training
```bash
    git clone https://github.com/mmmwembe/cards-training.git
```
F) Download test images and labels.txt to /tmp
```bash
    cp cards-training/images/real-test/* /tmp/
    cp cards-training/labels/labels.txt /tmp/
```
G) Run label_image.py
```bash
   python3 label_image.py
```

B) Modify label_image.py as below:

```bash
# Before
import tensorflow as tf
# After 
import tflite_runtime.interpreter as tflite

```
C) Run label_image.py

# Output Arrays from model predictions
# TFLite_Detection_PostProcess =Bounding Boxes
# TFLite_Detection_PostProcess:1=Detected Classes
# TFLite_Detection_PostProcess:2=Classes Precision
# TFLite_Detection_PostProcess:3=Number of Detected Objects