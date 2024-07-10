const ort = require("onnxruntime-node");
const express = require('express');
const multer = require("multer");
const sharp = require("sharp");
const fs = require("fs");

/**
 * Main function that setups and starts a
 * web server on port 8080
 */
function main() {
    const app = express();
    const upload = multer();

    /**
     * The site root handler. Returns content of index.html file.
     */
    app.get("/", (req, res) => {
        res.sendFile(__dirname + "/index.html");
    });

    /**
     * The handler of /detect endpoint that receives uploaded
     * image file, passes it through YOLOv8 object detection network and returns
     * an array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..] as a JSON
     */
    app.post('/detect', upload.single('image_file'), async function (req, res) {
        const boxes = await detect_objects_on_image(req.file.buffer);
        res.json(boxes);
    });

    app.listen(8080, () => {
        console.log(`Server is listening on port 8080`);
    });
}

/**
 * Function receives an image, passes it through YOLOv8 neural network
 * and returns an array of detected objects and their bounding boxes
 * @param buf Input image body
 * @returns Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
 */
async function detect_objects_on_image(buf) {
    const [input, img_width, img_height] = await prepare_input(buf);
    const output = await run_model(input);
    return process_output(output, img_width, img_height);
}

/**
 * Function used to convert input image to tensor,
 * required as an input to YOLOv8 object detection
 * network.
 * @param buf Content of uploaded file
 * @returns Array of pixels
 */
async function prepare_input(buf) {
    const img = sharp(buf);
    const md = await img.metadata();
    const [img_width, img_height] = [md.width, md.height];
    const pixels = await img.removeAlpha()
        .resize({ width: 640, height: 640, fit: 'fill' })
        .raw()
        .toBuffer();
    const red = [], green = [], blue = [];
    for (let index = 0; index < pixels.length; index += 3) {
        red.push(pixels[index] / 255.0);
        green.push(pixels[index + 1] / 255.0);
        blue.push(pixels[index + 2] / 255.0);
    }
    const input = [...red, ...green, ...blue];
    return [input, img_width, img_height];
}

/**
 * Function used to pass provided input tensor to YOLOv8 neural network and return result
 * @param input Input pixels array
 * @returns Raw output of neural network as a flat array of numbers
 */
async function run_model(input) {
    const model = await ort.InferenceSession.create("yolov8m.onnx");
    input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
    const outputs = await model.run({ images: input });
    return outputs["output0"].data;
}

/**
 * Function used to convert RAW output from YOLOv8 to an array of detected objects.
 * Each object contain the bounding box of this object, the type of object and the probability
 * @param output Raw output of YOLOv8 network
 * @param img_width Width of original image
 * @param img_height Height of original image
 * @returns Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
 */
function process_output(output, img_width, img_height) {
    let boxes = [];
    for (let index = 0; index < 8400; index++) {
        const [class_id, prob] = [...Array(80).keys()]
            .map(col => [col, output[8400 * (col + 4) + index]])
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);
        if (prob < 0.5) {
            continue;
        }
        const label = yolo_classes[class_id];
        const xc = output[index];
        const yc = output[8400 + index];
        const w = output[2 * 8400 + index];
        const h = output[3 * 8400 + index];
        const x1 = (xc - w / 2) / 640 * img_width;
        const y1 = (yc - h / 2) / 640 * img_height;
        const x2 = (xc + w / 2) / 640 * img_width;
        const y2 = (yc + h / 2) / 640 * img_height;
        boxes.push([x1, y1, x2, y2, label, prob]);
    }

    boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0], box) < 0.7);
    }
    return result;
}

/**
 * Function calculates "Intersection-over-union" coefficient for specified two boxes
 * https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
 * @param box1 First box
 * @param box2 Second box
 * @returns Intersection-over-union value
 */
function iou(box1, box2) {
    const intersect = intersection(box1, box2);
    const union_area = union(box1, box2, intersect);
    return intersect / union_area;
}

/**
 * Function calculates the union area of two bounding boxes
 * @param box1 First box
 * @param box2 Second box
 * @param intersect Area of intersection between two bounding boxes
 * @returns Union area of two boxes
 */
function union(box1, box2, intersect) {
    const [x11, y11, x12, y12] = box1;
    const [x21, y21, x22, y22] = box2;
    const area1 = (x12 - x11) * (y12 - y11);
    const area2 = (x22 - x21) * (y22 - y21);
    return area1 + area2 - intersect;
}

/**
 * Function calculates the intersection area of two bounding boxes
 * @param box1 First box
 * @param box2 Second box
 * @returns Intersection area of two boxes
 */
function intersection(box1, box2) {
    const [x11, y11, x12, y12] = box1;
    const [x21, y21, x22, y22] = box2;
    const xi1 = Math.max(x11, x21);
    const yi1 = Math.max(y11, y21);
    const xi2 = Math.min(x12, x22);
    const yi2 = Math.min(y12, y22);
    const inter_width = Math.max(xi2 - xi1, 0);
    const inter_height = Math.max(yi2 - yi1, 0);
    return inter_width * inter_height;
}

// YOLOv8 class labels
const yolo_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
];

main();
