using System;
using System.Collections.Generic;
using System.Web.Http;
using System.Threading.Tasks;
using TNCAnimalLabelWebAPI.CNTK;
using TNCAnimalLabelWebAPI.Models;


namespace TNCAnimalLabelWebAPI.Controllers
{
    public class PredictionController : ApiController
    {
        public static CNTKModel CNTK_Model = new CNTKModel("");

        // GET: api/Prediction
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET: api/Prediction/5
        public string Get(int id)
        {
            return "value";
        }

        // POST: api/Prediction
        public async Task<IHttpActionResult> Post([FromBody]string Url)
        {
            // parameter binding: https://docs.microsoft.com/en-us/aspnet/web-api/overview/formats-and-model-binding/parameter-binding-in-aspnet-web-api

            // retrieve the "Prediction-key" header.   
            string prediction_key;
            try
            {
                string[] headers = (string[])Request.Headers.GetValues("Prediction-key");
                prediction_key = headers[0];
            }
            catch (Exception ex)
            {
                // if the prediction_key doesn't exist, simply set it to default value - "".
                prediction_key = "";
            }


            try
            {
                var predictionResult = await CNTK_Model.EvaluateCustomDNN("", "", prediction_key, Url);
                return Ok(predictionResult);
            }
            catch (Exception ex)
            {
                return Ok(ex.ToString());
            }
        }

        // POST api/Prediction/?iterationId={string}&application={string}
        public async Task<IHttpActionResult> Post(string iterationId, string application, [FromBody]string Url)
        {
            // retrieve the "Prediction-key" header.   
            string prediction_key;
            try
            {
                string[] headers = (string[])Request.Headers.GetValues("Prediction-key");
                prediction_key = headers[0];
            }
            catch (Exception ex)
            {
                // if the prediction_key doesn't exist, simply set it to default value - "".
                prediction_key = "";
            }


            try
            {
                ImagePredictionResultModel predictionResult = await CNTK_Model.EvaluateCustomDNN(iterationId, application, prediction_key, Url);
                return Ok(predictionResult);
            }
            catch (Exception ex)
            {
                return Ok(ex.ToString());
            }
        }


        // PUT: api/Prediction/5
        public void Put(int id, [FromBody]string value)
        {

        }

        // DELETE: api/Prediction/5
        public void Delete(int id)
        {
        }

    }
}
