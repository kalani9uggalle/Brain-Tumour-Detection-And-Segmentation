//========================================================================
// Drag and drop image handling
//========================================================================


var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result1");
var predResult_middle = document.getElementById("pred-result2");

var loader = document.getElementById("loader");
var imageDisplay_data = document.getElementById("image-display2");
var model_first = undefined;
var model_second = undefined;
var model_third  = undefined;
//========================================================================
// Main button events
//========================================================================

// Method 1: Canvas - Display pred image using tf.toPixels
        //=====================================================
        

const canvas  = document.createElement('canvas');
var div = document.getElementById("img");
  

async function initialize() {
    model_first = await tf.loadLayersModel('/weights_first/model.json');
    model_second = await tf.loadLayersModel('/weights_second/model.json');
    model_third = await tf.loadLayersModel('/weights_third/model.json');
}

async function predict() {
  // action for the submit button
  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  let tensorImg = tf.browser.fromPixels(imagePreview)
	.resizeNearestNeighbor([224,224]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
	;

 
  prediction = await model_first.predict(tensorImg).data();
  // console.log (prediction[0]);
  if (prediction[1]==1) {
      console.log ("I think it's a brain image");
      predResult.innerHTML = "I think it's a brain image";
      prediction_middle = await model_second.predict(tensorImg).data();
      if (prediction_middle >=  0.5) {
        predResult_middle.innerHTML = "I think it's a tumor brain image";
        console.log ("I think it's a tumor brain image");

        let orig = tf.browser.fromPixels(imagePreview)
        let resized_img = tf.image.resizeNearestNeighbor(orig ,[256,256])
        let tensor_gray = resized_img .mean(2,true).toFloat().div(tf.scalar(255))
        // console.log(tensor_gray.shape)
       //get image as RGB //(256,256,3)
      
        let exanded_gray = tensor_gray.expandDims(0)
        
        // console.log(exanded_gray.shape)
     

        // //image for prediction
        // const predict_img = orig_resized
        // .mean(2)
        // .toFloat()
        // .div(255)
        // .expandDims(0)
        // .expandDims(-1);
       
        // console.log( predict_img)
        // Get the color image
        // var color_image=  tensorImg_1.resizeNearestNeighbor([256,256]);
        // var color_image= tensorImg.resizeNearestNeighbor([256,256]);
        // User submitted image
        // var orig_image = tf.browser.fromPixels(imagePreview).resizeNearestNeighbor([256,256]);
        // This is how we print info on tensors to the console:
        // verbose can be left out e.g. tensor.print()
        const verbose = true;
        // tensor.print(verbose);
        
       
	      
        const prediction_last = await model_third.predict(exanded_gray).squeeze().data();
        console.log("prediction done...")
        // console.log(prediction_last.shape)
        
        // convert typed array to a javascript array
        const pred_array = Array.from(prediction_last); // JS Code with js array
        // console.log(pred_array)
        tf.dispose(prediction_last); // dispose 
        
        
      
        // threshold the predictions
        var i;
   
        var num;
        for (i = 0; i < pred_array.length; i++) { 
          
          num = pred_array[i]*255;
          
          if (num <= 110) {   // <-- Set the threshold here
            pred_array[i] = 0;
            
          } else {
       
            pred_array[i] = 255;
            
            
          }
          
        }
        // console.log(c)
        console.log("preds array")
        // console.log('Input Image shape: ',  pred_array);  
        // convert js array to a tensor
        pred_tensor = tf.tensor1d(pred_array, 'int32');
        // console.log('Input Image shape: ',  pred_tensor.shape);  
        
        // reshape the pred tensor
        pred_tensor = pred_tensor.reshape([256,256]).div(tf.scalar(255)).expandDims(-1);
        // console.log('pred_tensor shape: ',  pred_tensor.shape);
        // console.log(' orig_image  shape: ',   orig_image.shape);


        // resize pred_tensor
        // pred_tensor = pred_tensor.resizeNearestNeighbor([orig_image.shape[0], orig_image.shape[1]]);
        
        // reshape the input image tensor
        //input_img_tensor = tensor.reshape([128,128,3]);
         
        // append the tensor to the input image to create a 4th alpha channel --> shape [128,128,4]
        rgba_tensor = tf.concat([tensor_gray, pred_tensor], axis=-1);
       
        
        // resize all images. rgb_tensor is the segmented image.
        rgba_tensor = rgba_tensor.resizeNearestNeighbor([256, 256]).mean(2);
        // orig_image = orig_image.resizeNearestNeighbor([256, 256]);
        // color_image = color_image.resizeNearestNeighbor([256, 256]);
        
        console.log(rgba_tensor.shape)

        canvas.width = rgba_tensor.shape.width
        canvas.height = rgba_tensor.shape.height
        
        await tf.browser.toPixels(rgba_tensor , canvas); 
        
        div.appendChild(canvas);
        show(canvas)
        // Convert the tensor to an image 
        
        
        

        // tf.browser.toPixels(orig_image, canvas3);
        // tf.browser.toPixels(color_image, canvas4);
      }
      else{
        predResult_middle.innerHTML = "I think it's a normal brain image";
        console.log ("I think it's a normal brain image");
      }
      show(predResult_middle)
     


  } else if (prediction[0]==1) {
      console.log ( "I think it's an animal");
      predResult.innerHTML ="I think it's an animal";

  } else if (prediction[2]==1) {
    console.log ("I think it's a non-brain image");
    predResult.innerHTML = "I think it's a non-brain image";
  } else {
      predResult.innerHTML = "This is Something else";
     
  }
  show(predResult)
 
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";
  predResult_middle.innerHTML = "";

 


  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);
  hide(canvas)

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  // show the preview of the image
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    imageDisplay.classList.remove("loading");

    displayImage(reader.result, "image-display");
  };
}

//========================================================================
// Helper functions
//========================================================================

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}

initialize();