const STATUS = document.getElementById("status");
const WEBCAM = document.getElementById("webcam");
const ENABLE = document.getElementById("enableCam");
const RESET = document.getElementById("reset");
const TRAIN = document.getElementById("train");
const SNAPSHOTS = document.getElementById("snapshots");
const CLASS_CONTROLS = document.getElementById("classControls");
const ADD_CLASS = document.getElementById("addClass");
let videoFrameAsTensor;
const MN_INPUT_WIDTH = 224;
const MN_INPUT_HEIGHT = 224;
const STOP_DATA_GATER = -1;
const CLASS_NAMES = ["Class 1", "Class 2"];
let model;

let mobilenet = undefined;
let dataGaterState = STOP_DATA_GATER;
let videoPlaying = false;
let trainingDataInput = [];
let trainingDataOutput = [];
let examplesCount = [];
let predict = false;

ENABLE.addEventListener("click", enableCam);
TRAIN.addEventListener("click", trainAndPredict);
RESET.addEventListener("click", reset);
ADD_CLASS.addEventListener("click", addClass);

let dataCollectorsButton = [];
let editClassButtons = [];

renderClassControls();
createSnapshotRows();
createModel();

function createSnapshotRows() {
  SNAPSHOTS.innerHTML = "";
  for (let i = 0; i < CLASS_NAMES.length; i++) {
    appendSnapshotRow(i);
  }
}

function appendSnapshotRow(index) {
  const row = document.createElement("div");
  row.className = "snapshot-row";
  row.id = `snapshot-row-${index}`;

  const title = document.createElement("div");
  title.className = "snapshot-row-title";
  title.id = `snapshot-title-${index}`;
  title.innerText = CLASS_NAMES[index];

  row.appendChild(title);
  SNAPSHOTS.appendChild(row);
}

function renderClassControls() {
  CLASS_CONTROLS.innerHTML = "";

  for (let i = 0; i < CLASS_NAMES.length; i++) {
    const group = document.createElement("div");
    group.className = "data-collector-group";
    group.setAttribute("data-class-index", i);

    const gather = document.createElement("button");
    gather.className = "dataCollector";
    gather.type = "button";
    gather.setAttribute("data-1hot", i);
    gather.setAttribute("data-name", CLASS_NAMES[i]);
    gather.innerText = `Gather ${CLASS_NAMES[i]}`;
    gather.addEventListener("pointerdown", startGatheringForClass);
    gather.addEventListener("pointerup", stopGathering);
    gather.addEventListener("pointerleave", stopGathering);
    gather.addEventListener("pointercancel", stopGathering);
    gather.addEventListener("touchcancel", stopGathering);
    gather.addEventListener("touchend", stopGathering);

    const edit = document.createElement("button");
    edit.className = "editClassName";
    edit.type = "button";
    edit.setAttribute("data-class-index", i);
    edit.innerText = "✏️";
    edit.addEventListener("click", renameClassName);

    const remove = document.createElement("button");
    remove.className = "removeClassName";
    remove.type = "button";
    remove.setAttribute("data-class-index", i);
    remove.innerText = "🗑️";
    remove.disabled = CLASS_NAMES.length <= 2;
    remove.addEventListener("click", removeClass);

    group.appendChild(gather);
    group.appendChild(edit);
    group.appendChild(remove);
    CLASS_CONTROLS.appendChild(group);
  }

  ADD_CLASS.disabled = CLASS_NAMES.length >= 5;

  dataCollectorsButton = document.querySelectorAll("button.dataCollector");
  editClassButtons = document.querySelectorAll("button.editClassName");
}

function renameClassName(event) {
  const button = event.currentTarget;
  const index = parseInt(button.getAttribute("data-class-index"), 10);
  const currentName = CLASS_NAMES[index] || `Class ${index + 1}`;
  const newName = prompt("Enter new class name:", currentName);

  if (!newName || !newName.trim()) {
    return;
  }

  const trimmedName = newName.trim();
  CLASS_NAMES[index] = trimmedName;

  const gatherButton = document.querySelector(`button.dataCollector[data-1hot="${index}"]`);
  if (gatherButton) {
    gatherButton.setAttribute("data-name", trimmedName);
    gatherButton.innerText = `Gather ${trimmedName}`;
  }

  const title = document.getElementById(`snapshot-title-${index}`);
  if (title) {
    title.innerText = trimmedName;
  }

  if (examplesCount[index] !== undefined) {
    updateStatusCounts();
  }
}

function createModel() {
  if (model) {
    model.dispose();
  }

  model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
  );

  model.compile({
    optimizer: "adam",
    loss:
      CLASS_NAMES.length === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
}

function addClass() {
  if (CLASS_NAMES.length >= 5) {
    alert("Maximum 5 classes allowed.");
    return;
  }

  const nextIndex = CLASS_NAMES.length;
  CLASS_NAMES.push(`Class ${nextIndex + 1}`);
  examplesCount[nextIndex] = 0;
  renderClassControls();
  appendSnapshotRow(nextIndex);
  createModel();
  updateStatusCounts();
}

function removeClass(event) {
  const button = event.currentTarget;
  const removeIndex = parseInt(button.getAttribute("data-class-index"), 10);

  if (CLASS_NAMES.length <= 2) {
    alert("At least 2 classes are required.");
    return;
  }

  CLASS_NAMES.splice(removeIndex, 1);
  examplesCount.splice(removeIndex, 1);

  for (let i = trainingDataInput.length - 1; i >= 0; i--) {
    if (trainingDataOutput[i] === removeIndex) {
      trainingDataInput[i].dispose();
      trainingDataInput.splice(i, 1);
      trainingDataOutput.splice(i, 1);
    } else if (trainingDataOutput[i] > removeIndex) {
      trainingDataOutput[i]--;
    }
  }

  if (dataGaterState === removeIndex) {
    dataGaterState = STOP_DATA_GATER;
  } else if (dataGaterState > removeIndex) {
    dataGaterState--;
  }

  const removedRow = document.getElementById(`snapshot-row-${removeIndex}`);
  if (removedRow) {
    removedRow.remove();
  }

  // Renumber snapshot rows
  for (let i = removeIndex; i < CLASS_NAMES.length; i++) {
    const row = document.getElementById(`snapshot-row-${i + 1}`);
    if (row) {
      row.id = `snapshot-row-${i}`;
      const title = row.querySelector('.snapshot-row-title');
      if (title) {
        title.id = `snapshot-title-${i}`;
      }
    }
  }

  renderClassControls();
  createModel();
  updateStatusCounts();
}

async function loadMobileNetModel() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });

  tf.tidy(() => {
    let answer = mobilenet.predict(
      tf.zeros([1, MN_INPUT_HEIGHT, MN_INPUT_WIDTH, 3])
    );
    console.log(answer.shape); // should be [1, 1024]
  });

  STATUS.innerHTML = "Model Loaded Successfully";
}

loadMobileNetModel();

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = {
      video: true,
      width: 640,
      height: 480
    };
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      WEBCAM.srcObject = stream;
      WEBCAM.play();
      WEBCAM.addEventListener("loadeddata", function () {
        videoPlaying = true;
        ENABLE.classList.add("removed");
      });
    });
  } else {
    console.warn("No Cams");
  }
}

function startGatheringForClass(event) {
  event.preventDefault();
  const classNumber = parseInt(event.currentTarget.getAttribute("data-1hot"), 10);
  dataGaterState = classNumber;
  dataGatterLoop();
}

function stopGathering() {
  dataGaterState = STOP_DATA_GATER;
}

function dataGatterLoop() {
  if (videoPlaying && dataGaterState !== STOP_DATA_GATER) {
    let imageFeatures = tf.tidy(function () { videoFrameAsTensor = tf.browser.fromPixels(WEBCAM);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MN_INPUT_HEIGHT, MN_INPUT_WIDTH],
        true
      );
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
    trainingDataInput.push(imageFeatures);
    trainingDataOutput.push(dataGaterState);
    captureSnapshot(dataGaterState);
    console.log(tf.stack(trainingDataInput), trainingDataOutput);
    if (examplesCount[dataGaterState] === undefined) {
      examplesCount[dataGaterState] = 0;
    }
    examplesCount[dataGaterState]++;
    STATUS.innerText = "";
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerHTML +=
        CLASS_NAMES[n] + " data count: " + examplesCount[n] + " - ";
    }
    window.requestAnimationFrame(dataGatterLoop);
  }
}

function captureSnapshot(classIndex) {
  const row = document.getElementById(`snapshot-row-${classIndex}`);
  if (!row) {
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = MN_INPUT_WIDTH;
  canvas.height = MN_INPUT_HEIGHT;
  canvas.className = "snapshot-preview";

  const ctx = canvas.getContext("2d");
  ctx.drawImage(WEBCAM, 0, 0, MN_INPUT_WIDTH, MN_INPUT_HEIGHT);
  row.appendChild(canvas);
}

async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInput, trainingDataOutput);
  let outputAsTensor = tf.tensor1d(trainingDataOutput, "int32");
  let oneHotOutputs = tf.oneHot(outputAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInput);
  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress }
  });
  outputAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
  predictLoop();
  console.log(results);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      let videoFrameAsTensor = tf.browser.fromPixels(WEBCAM).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MN_INPUT_HEIGHT, MN_INPUT_WIDTH],
        true
      );
      let imageFeature = mobilenet.predict(resizedTensorFrame.expandDims());
      let prediction = model.predict(imageFeature).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      STATUS.innerText = "Prediction " + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + ' %';
    });
    window.requestAnimationFrame(predictLoop);
  }
}

function reset(){
  predict = false;
  dataGaterState = STOP_DATA_GATER;

  for(let i = 0; i < trainingDataInput.length; i++){
    trainingDataInput[i].dispose();
  }
  trainingDataInput.splice(0);
  trainingDataOutput.splice(0);
  examplesCount.splice(0);
  SNAPSHOTS.innerHTML = '';
  createSnapshotRows();
  STATUS.innerText = 'No Data Collected';
  console.log('Tensors in memory ' + tf.memory().numTensors);
}

