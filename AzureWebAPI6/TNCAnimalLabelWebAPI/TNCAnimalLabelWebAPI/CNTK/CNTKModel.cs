using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using Microsoft.VisualBasic.FileIO;
using System.Net.Http;
using System.Drawing;
using TNCAnimalLabelWebAPI.Models;
using CNTK;
using CNTKImageProcessing;


namespace TNCAnimalLabelWebAPI.CNTK
{
    public class CNTKModel
    {
        private List<ModelClassLabelID> ClassLabelIDs { get; set; }
        private ConcurrentBag<Function> modelPool = new ConcurrentBag<Function>();
        private readonly Function _basemodel;
        private readonly int _max_allowed_stored_models;
        private readonly string _model_name;
        private readonly HttpClient _httpClient;

        public CNTKModel(string modelName)
        {
            // if no modelName is provided, fall-back to default 
            _model_name = string.IsNullOrWhiteSpace(modelName) ? "TNC_ResNet18_ImageNet_CNTK" : modelName;

            string domainBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;

            // load class labels and IDs.
            string classLableMapFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\" + _model_name + ".csv");
            this.ClassLabelIDs = this.LoadClassLabelAndIDsFromCSV(classLableMapFilePath);

            DeviceDescriptor device = DeviceDescriptor.CPUDevice;

            // Load the CNTK model.
            // This example requires the ResNet20_CIFAR10_CNTK.model.
            // The model can be downloaded from <see cref="https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model"/>
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            // string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\ResNet20_CIFAR10_CNTK.model");
            string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\" + _model_name + ".model");
            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
            }

            _httpClient = new HttpClient();
            _basemodel = Function.Load(modelFilePath, device);
            modelPool.Add(_basemodel.Clone());
        }

        private async Task<Bitmap> GetImageFromUrl(string imageUrl)
        {
            // Retrieve the image file.
            //Bitmap bmp = new Bitmap(Bitmap.FromFile(imageUrl));
            var imageStream = await _httpClient.GetStreamAsync(imageUrl);
            var bmp = new Bitmap(Image.FromStream(imageStream));
            return bmp;
        }

        private Function GetModelFromBag()
        {
            modelPool.TryTake(out var tfun);
            return tfun ?? _basemodel.Clone();
        }

        private bool ReturnModelToBag(Function cntkFunction)
        {
            if (modelPool.Count >= _max_allowed_stored_models) return false;
            modelPool.Add(cntkFunction);
            return true;
        }
        
        public async Task<ImagePredictionResultModel> EvaluateCustomDNN(string modelName, string iterationID, string application, string imageUrl)
        //public ImagePredictionResultModel EvaluateCustomDNN(string imageUrl)
        {


            Function modelFunc = GetModelFromBag();

            DeviceDescriptor device = DeviceDescriptor.CPUDevice;

            // Get input variable. The model has only one single input.
            Variable inputVar = modelFunc.Arguments.Single();

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            int imageWidth = inputShape[0];
            int imageHeight = inputShape[1];
            int imageChannels = inputShape[2];
            int imageSize = inputShape.TotalSize;

            // Get output variable
            Variable outputVar = modelFunc.Output;

            var inputDataMap = new Dictionary<Variable, Value>();
            var outputDataMap = new Dictionary<Variable, Value>();

            // Retrieve the image file.
            //Bitmap bmp = new Bitmap(Bitmap.FromFile(imageUrl));
            Bitmap bmp = await GetImageFromUrl(imageUrl);

            var resized = bmp.Resize(imageWidth, imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create output data map
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            ReturnModelToBag(modelFunc);

            // Get evaluate result as dense output
            var outputVal = outputDataMap[outputVar];
            var outputData = outputVal.GetDenseData<float>(outputVar);
            float[] softmax_vals = ActivationFunctions.Softmax(outputData[0]);

            if (softmax_vals.Length != this.ClassLabelIDs.Count)
            {
                throw new Exception(_model_name + " class label mapping file and CNTK model file have different number of classes.");
            }


            // construct a ImagePredictionResultModel.    "class name": prediction of the class.
            ImagePredictionResultModel predictionResult = new ImagePredictionResultModel
            {
                Id = "TNC100",
                Project = "TNCAnimalLabel",
                Iteration = iterationID,
                Created = DateTime.Now,
                Predictions = new List<Prediction>()
            };

            int i = 0;
            for (; i < (softmax_vals.Length); i++)
            {
                var prediction = new Prediction
                {
                    TagId = this.ClassLabelIDs[i].ID,
                    Tag = this.ClassLabelIDs[i].Label,
                    Probability = softmax_vals[i]
                };
                predictionResult.Predictions.Add(prediction);
            }

            return predictionResult;
        }

        private List<ModelClassLabelID> LoadClassLabelAndIDsFromCSV(string classLableMapFilePath)
        {
            if (!File.Exists(classLableMapFilePath))
            {
                throw new FileNotFoundException(classLableMapFilePath, string.Format("Error: The model class label mapping file '{0}' does not exist.", classLableMapFilePath));
            }

            List<ModelClassLabelID> modelLableIDs = new List<ModelClassLabelID>();

            using (TextFieldParser csvParser = new TextFieldParser(classLableMapFilePath))
            {
                csvParser.CommentTokens = new string[] { "#" };
                csvParser.SetDelimiters(new string[] { "," });
                csvParser.HasFieldsEnclosedInQuotes = true;

                // Skip the row with the column names
                csvParser.ReadLine();

                while (!csvParser.EndOfData)
                {
                    // Read current line fields, pointer moves to the next line.
                    string[] fields = csvParser.ReadFields();

                    ModelClassLabelID model_label_id = new ModelClassLabelID();
                    model_label_id.Label = fields[0];
                    model_label_id.ID = fields[1];
                    modelLableIDs.Add(model_label_id);
                }
            }

            return modelLableIDs;
        }




    }
}