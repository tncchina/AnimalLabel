using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Text;

using TNCAnimalLabelWebAPI.Models;
using CNTK;
using CNTKImageProcessing;

namespace TNCAnimalLabelWebAPI.Controllers
{
    public class PredictionController : ApiController
    {
        //public IHttpActionResult GetPrediction(string id)
        public async Task<IHttpActionResult> GetPrediction(string id)
        {
            try
            {
                string imageURL = "https://qqq.blob.core.windows.net/mycontainer/IMAG0005.JPG";
                //ImagePredictionResultModel predictionResult = this.EvaluateCustomDNN(@"C:\TNC_CNTK\data\test\IMAG0002.JPG");
                //ImagePredictionResultModel predictionResult = await this.EvaluateCustomDNN(@"C:\TNC_CNTK\data\test\IMAG0002.JPG");
                ImagePredictionResultModel predictionResult = await this.EvaluateCustomDNN(imageURL);

                return Ok(predictionResult);
            }
            catch (Exception ex)
            {
                return Ok(ex.ToString()) ;
            }
        }

        public async Task<ImagePredictionResultModel> EvaluateCustomDNN(string imageUrl)
        //public ImagePredictionResultModel EvaluateCustomDNN(string imageUrl)
        {
            string domainBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;
            string[] class_labels = new string[] { "空", "家牛", "人", "松鼠", "鼠", "猕猴", "血雉", "鸟类", "滇金丝猴", "麂属",
                "家狗", "兽类", "黄喉貂", "山羊", "白腹锦鸡", "豹猫", "赤麂", "红腹角雉", "黑颈长尾雉", "黄鼬", "亚洲黑熊", "绵羊",
                "野兔", "家羊", "鼯鼠", "鬣羚", "白顶溪鸲", "黄嘴山鸦", "家马", "黑顶噪鹛", "隐纹花鼠", "花面狸", "黑熊", "豪猪",
                "啄木鸟", "小麂", "鼯鼠属", "白点噪鹛", "长尾地鸫", "眼纹噪鹛", "灰头小鼯鼠", "勺鸡" };

            // Load the model.
            // This example requires the ResNet20_CIFAR10_CNTK.model.
            // The model can be downloaded from <see cref="https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model"/>
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            // string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\ResNet20_CIFAR10_CNTK.model");
            string modelFilePath = Path.Combine(domainBaseDirectory, @"CNTK\Models\TNC_ResNet18_ImageNet_CNTK.model");
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

            // construct a ImagePredictionResultModel.    "class name": prediction of the class.
            ImagePredictionResultModel predictionResult = new ImagePredictionResultModel();
            predictionResult.Id = "tnc100";
            predictionResult.Project = "EvaluateCustomDNN";
            predictionResult.Iteration = "1.00";
            predictionResult.Created = DateTime.Now;
            predictionResult.Predictions = new List<Prediction>();

            int class_id = 0;
            for (; class_id < (softmax_vals.Length); class_id++)
            {
                Prediction prediction = new Prediction();
                prediction.TagId = class_id.ToString();
                prediction.Tag = class_labels[class_id];
                prediction.Probability = softmax_vals[class_id];
                predictionResult.Predictions.Add(prediction);
            }

            return predictionResult;
        }


    }
}
