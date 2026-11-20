# HemoPred

HemoPred is a web-based tool to predict hemoglobin A2 (HbA2) and fetal hemoglobin 
(HbF) from age and routine blood test features - complete blood counts (CBCs) and 
red blood cell (RBC) indices.

It uses regression models trained with Mixture of Experts (MoE) approach.  The 
models were trained on an Indian dataset of 5,314 individuals, obtained from 
Postgraduate Institute of Medical Education and Research (PGIMER), Chandigarh.

HemoPred also provides feature attribution using SHapley Additive exPlanations
(SHAP) algorithm, and assigns labels along with affinity score, based on the 
predicted HbA2 and HbF values. 

 **Cite as:**

>Bhattacharjee, S., Christensen, F., Maity, S., Sharma, P., Jamwal, M., Das, R., 
& Nielsen, I.E. (2026). HemoPred: A web application for hemoglobin A2 (HbA2) and 
fetal hemoglobin (HbF) prediction from routine blood features using Mixture of 
Experts and explainable artificial intelligence. *Computers in Biology and 
Medicine*, 219, 112077.<br/>
[https://doi.org/10.1016/j.compbiomed.2026.112077](https://doi.org/10.1016/j.compbiomed.2026.112077).

## Using the tool

HemoPred is available at: [https://aihemo.mp.aau.dk/hemopred](https://aihemo.mp.aau.dk/hemopred).

To know more about the dataset, methodology and the performance, please refer to 
the [About](https://aihemo.mp.aau.dk/hemopred/about.html) page. Please refer to 
the [Help](https://aihemo.mp.aau.dk/hemopred/help.html) page for understanding 
the inputs and outputs to the web application.

## Development

Python libraries used :

* numpy (Version `2.4.2`)
* scikit-learn (Version `1.5.0`)
* scipy (Version `1.17.0`)
* shap (Version `0.51.0`)
* MAPIE (Version `1.4.0`)
* joblib (Version `1.5.3`)
* matplotlib (Version `3.10.8`)

R libraries used :

* mcr (Version `1.3.3.1`)
* boot (Version `1.3-32`)
* dplyr (Version `1.2.1`)
* ggplot2 (Version `4.0.3`)
* ggpubr (Version `1.0.0`)
* patchwork (Version `1.3.2`)
* magick (Version `2.9.1`)

The web application was developed using PHP (Version `8.3`).

## Team
* **Sudipto Bhattacharjee** ([Email](mailto:sbha@mp.aau.dk))<br/>
  AI4OR group, Department of Materials and Production,<br/>
  Aalborg University, Aalborg, Denmark.<br/>
* **Frederik Christensen** ([Email](mailto:frch@mp.aau.dk))<br/>
  AI4OR group, Department of Materials and Production,<br/>
  Aalborg University, Aalborg, Denmark.<br/>
* **Samir Maity** ([Email](mailto:samirm@mp.aau.dk))<br/>
  AI4OR group, Department of Materials and Production,<br/>
  Aalborg University, Aalborg, Denmark.<br/>
* **Prashant Sharma** ([Email](mailto:sharma.prashant@pgimer.edu.in))<br/>
  Department of Hematology,<br/>
  Postgraduate Institute of Medical Education and Research (PGIMER), Chandigarh, India.
* **Manu Jamwal** ([Email](mailto:manujamwal25@gmail.com))<br/>
  Department of Hematology,<br/>
  Postgraduate Institute of Medical Education and Research (PGIMER), Chandigarh, India.
* **Reena Das** ([Email](mailto:das.reena@pgimer.edu.in))<br/>
  Department of Hematology,<br/>
  Postgraduate Institute of Medical Education and Research (PGIMER), Chandigarh, India.
* **Izabela Ewa Nielsen** ([Email](mailto:izabela@mp.aau.dk))<br/>
  AI4OR group, Department of Materials and Production,<br/>
  Aalborg University, Aalborg, Denmark.<br/>
  
*Please contact Izabela Ewa Nielsen regarding any further queries.*

*This tool is strictly for research use only. It should be used for medical purposes only by consulting with doctors.*
