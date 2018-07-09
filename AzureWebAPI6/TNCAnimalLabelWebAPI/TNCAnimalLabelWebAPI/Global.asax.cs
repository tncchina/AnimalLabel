using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Http;
using System.Web.Mvc;
using System.Web.Optimization;
using System.Web.Routing;

namespace TNCAnimalLabelWebAPI
{
    public class WebApiApplication : System.Web.HttpApplication
    {
        protected void Application_Start()
        {
            AreaRegistration.RegisterAllAreas();
            GlobalConfiguration.Configure(WebApiConfig.Register);
            FilterConfig.RegisterGlobalFilters(GlobalFilters.Filters);
            RouteConfig.RegisterRoutes(RouteTable.Routes);
            BundleConfig.RegisterBundles(BundleTable.Bundles);

            string pathValue = Environment.GetEnvironmentVariable("PATH");
            string domainBaseDir = AppDomain.CurrentDomain.BaseDirectory;
            string cntkPath = domainBaseDir + @"bin\";
            pathValue += ";" + cntkPath;
            Environment.SetEnvironmentVariable("PATH", pathValue);
        }
    }
}
