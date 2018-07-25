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
        public List<ModelClassLabelID> ClassLabelIDs { get; set; }
        public Function ModelFunc { get; set; }

    }
}