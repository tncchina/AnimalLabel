using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.VisualBasic.FileIO;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using System.Drawing;

using TNCAnimalLabelWebAPI.CNTK;
using TNCAnimalLabelWebAPI.Models;
using CNTK;
using CNTKImageProcessing;


namespace TNCAnimalLabelWebAPI.CNTK
{
    public class CNTKModel
    {
        private List<ModelClassLabelID> ClassLabelIDs { get; set; }

        public async Task<ImagePredictionResultModel> EvaluateCustomDNN(string iterationID, string application, string modelName, string imageUrl)
        //public ImagePredictionResultModel EvaluateCustomDNN(string imageUrl)
        {
            // if no modelName is provided, fall-back to default 
            if (modelName == "")
            {
                modelName = "TNC_ResNet18_ImageNet_CNTK";
            }

            string domainBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;

            // load class labels and IDs.
            string classLableMapFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\" + modelName + ".csv");
            this.ClassLabelIDs = this.LoadClassLabelAndIDsFromCSV(classLableMapFilePath);

            DeviceDescriptor device = DeviceDescriptor.CPUDevice;

            // Load the CNTK model.
            // This example requires the ResNet20_CIFAR10_CNTK.model.
            // The model can be downloaded from <see cref="https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model"/>
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            // string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\ResNet20_CIFAR10_CNTK.model");
            string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\" + modelName + ".model");
            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
            }

            Function modelFunc = Function.Load(modelFilePath, device);

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
            System.Net.Http.HttpClient httpClient = new HttpClient();
            Stream imageStream = await httpClient.GetStreamAsync(imageUrl);
            Bitmap bmp = new Bitmap(Bitmap.FromStream(imageStream));

            var resized = bmp.Resize(imageWidth, imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create output data map
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            var outputVal = outputDataMap[outputVar];
            var outputData = outputVal.GetDenseData<float>(outputVar);
            float[] softmax_vals = ActivationFunctions.Softmax(outputData[0]);

            if (softmax_vals.Length != this.ClassLabelIDs.Count)
            {
                throw new Exception(modelName + " class label mapping file and CNTK model file have different number of classes.");
            }


            // construct a ImagePredictionResultModel.    "class name": prediction of the class.
            ImagePredictionResultModel predictionResult = new ImagePredictionResultModel();
            predictionResult.Id = "TNC100";
            predictionResult.Project = "TNCAnimalLabel";
            predictionResult.Iteration = iterationID;
            predictionResult.Created = DateTime.Now;
            predictionResult.Predictions = new List<Prediction>();

            int i = 0;
            for (; i < (softmax_vals.Length); i++)
            {
                Prediction prediction = new Prediction();
                prediction.TagId = this.ClassLabelIDs[i].ID;
                prediction.Tag = this.ClassLabelIDs[i].Label;
                prediction.Probability = softmax_vals[i];
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