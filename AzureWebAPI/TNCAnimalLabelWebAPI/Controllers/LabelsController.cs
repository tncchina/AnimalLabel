using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using Swashbuckle.Swagger.Annotations;

using CNTK;
using CNTKImageProcessing;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Text;



namespace TNCAnimalLabelWebAPI.Controllers
{
    //[Route("api/labels")]
    public class LabelsController : ApiController
    {
        // GET api/<controller>
        //[HttpGet]
        [Route("api/labels")]
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/<controller>/5
        //[HttpGet("{id}"), Name = "GetID")]
        //[Route("api/labels/{id}")]
        //public string Get(int id)
        //{
        //    return "value";
        //}


        // GET api/<controller>/URL_to_image_file
        [Route("api/labels/{img_file_url_base64}")]
        public async Task<IEnumerable<string>> Get(string img_file_url_base64)
        {
            //return await this.EvaluateCustomDNN("http://3.bp.blogspot.com/-Mwr4UZALiA0/TWBt-3vFR8I/AAAAAAAAA4Y/0tXjI-NhVPM/s1600/j0262568.jpg");
            var encodedTextBytes = Convert.FromBase64String(img_file_url_base64);
            string img_file_url = Encoding.UTF8.GetString(encodedTextBytes);
            return await this.EvaluateCustomDNN(img_file_url);
            
        }


        // POST api/<controller>
        public void Post([FromBody]string value)
        {
        }

        // PUT api/<controller>/5
        public void Put(int id, [FromBody]string value)
        {
        }

        // DELETE api/<controller>/5
        public void Delete(int id)
        {
        }

        public async Task<string[]> EvaluateCustomDNN(string imageUrl)
        {
            string domainBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string workingDirectory = Environment.CurrentDirectory;
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;

            try
            {
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

                // // Image preprocessing to match input requirements of the model.
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
                List<string> o = new List<string>();

                // Get results from index 0 in output data since our batch consists of only one image
                //foreach (float f in outputData[0])
                foreach (float f in softmax_vals)
                {
                    o.Add( Math.Round(f, 5).ToString());
                }

                return o.ToArray<string>();
            }
            catch (Exception ex)
            {
                return new string[] { string.Format("domainBase directory {0}, workingDirectory {1}, exception details: {2}.", domainBaseDirectory, workingDirectory, ex.ToString()) };
            }
        }
    }
}