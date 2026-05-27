<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Home - HemoPred</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type = "text/javascript" src = "js/main.js"></script>
        <style>
            #sec-intro {
               width:47%;
               float:left;
               padding-top:50px;
            }
            #sec-form {
                width:47%;
                float:right;
            }
        </style>
    </head>
    <body>
        <div class = "section_header">
            <center><p class="title">HemoPred - Prediction of HbA2 and HbF from CBCs and RBC indices</p></center>
        </div>

        <div class = "section_menu">
            <center>
            <table cellpadding="3px">
                <tr class="nav">
                    <td class="nav"><a href="#" class="active">Home</a></td>
                    <td class="nav"><a href="about.html" class="side_nav">About</a></td>
                    <td class="nav"><a href="help.html" class="side_nav">Help</a></td>
                    <td class="nav"><a href="team.html" class="side_nav">Team</a></td>
                </tr>
            </table>
            </center>
        </div>

        <!--<div class = "section_left"></div>-->
        <div class = "section_middle">
            <div id="sec-intro">
                <p style="font-size: 1.15em;">
                    <b><font color="#800000">HemoPred</font></b> is a web-based tool to predict
                    hemoglobin A2 (HbA2) and fetal hemoglobin (HbF) from routine blood test
                    features, including complete blood counts (CBCs) and red blood cell (RBC)
                    indices. It uses regression models trained with Mixture of Experts (MoE)
                    approach. The MoE model for HbA2 regression uses two Gradient Boosted
                    Regressors (GBR) and a Support Vector Regressor (SVR) as experts.
                    Similarly, the MoE model for HbF regression uses two Quantile Regressors (QR)
                    and a SVR as experts. A Random Forest model was used for the gating mechanism.
                    The models were trained on an Indian dataset of 5,314 individuals.
                </p>
                <p style="font-size: 1.15em;">
                    HemoPred also assigns labels along with affinity score, based on the predicted
                    HbA2 and HbF values, using a Nearest Centroid model.
                </p>
                <p style="font-size: 1.15em;">
                    To know more about HemoPred, go to <a style="color:#800000;" href="about.html">About</a>
                    page. For help, please refer to <a style="color:#800000;" href="help.html">Help</a> page.
                </p>
            </div>
            <div id="sec-form">
                <center><h1 style="margin-top:5px; margin-bottom:5px;">Inputs</h1></center>
                <form name="input_form" method="post" action="prediction.php">
                    <center>
                        <input type="button" style="width:200px; margin:5px;" value="Insert example data 1" onclick="addDemoData(0)" />
                        <input type="button" style="width:200px; margin:5px;" value="Insert example data 2" onclick="addDemoData(1)" />
                        <input type="button" style="width:200px; margin:5px;" value="Insert example data 3" onclick="addDemoData(2)" />
                    </center>
                    <br/>
                    <table class="summary">
                        <tr>
                            <th style="width:60%; text-align: left;">Age (years)</th>
                            <td><input type="number" class="full" id="v1" name="age" min="18" max="90" step="1" value="" placeholder="Enter Age" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Hemoglobin - Hb (g/dL)</th>
                            <td><input type="number" class="full" id="v4" name="hb" min="1.0" max="20.0" step="0.1" value="" placeholder="Enter Hb" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Hematocrit - HCT (%)</th>
                            <td><input type="number" class="full" id="v5" name="hct" min="10.0" max="70.0" step="0.1" value="" placeholder="Enter HCT" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Red Blood Cell - RBC count (&times; 10<sup>12</sup> /L)</th>
                            <td><input type="number" class="full" id="v3" name="rbc" min="0.50" max="9.00" step="0.01" value="" placeholder="Enter RBC count" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Mean Corpuscular Volume - MCV (fL)</th>
                            <td><input type="number" class="full" id="v6" name="mcv" min="50.0" max="150.0" step="0.1" value="" placeholder="Enter MCV" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Mean Corpuscular Hemoglobin - MCH (pg)</th>
                            <td><input type="number" class="full" id="v7" name="mch" min="10.0" max="50.0" step="0.1" value="" placeholder="Enter MCH" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Mean Corpuscular Hemoglobin Concentration - MCHC (g/dL)</th>
                            <td><input type="number" class="full" id="v8" name="mchc" min="20.0" max="50.0" step="0.1" value="" placeholder="Enter MCHC" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Red cell Distribution Width - RDW (%)</th>
                            <td><input type="number" class="full" id="v9" name="rdw" min="10.0" max="50.0" step="0.1" value="" placeholder="Enter RDW-CV" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">White Blood Cell - WBC count (&times; 10<sup>9</sup> /L)</th>
                            <td><input type="number" class="full" id="v2" name="wbc" min="0.2" max="70.0" step="0.1" value="" placeholder="Enter WBC count" required /></td>
                        </tr>
                        <tr>
                            <th style="width:60%; text-align: left;">Platelet count (&times; 10<sup>9</sup> /L)</th>
                            <td><input type="number" class="full" id="v10" name="plt" min="1.0" max="1500" step="1" value="" placeholder="Enter Platelet count" required /></td>
                        </tr>
                    </table>
                    <p style="margin:10px 0 5px 0; padding:0;"><i>N.B. - All fields are mandatory.</i></p>
                    <input type = "hidden" id = "query" name = "query" value = "0" />

<!--                     <br/> -->
                    <center>
                        <input type="submit" style="width:70px;margin:5px;font-weight:em;" value="Submit" />
                        <input type="reset" style="width:70px;margin:5px;;font-weight:em;" value="Reset" />
                    </center>
                </form>
            </div>
            <div style="clear:both;"></div>

            <hr/>
            <p style="font-size:0.9em; text-align:center; margin-bottom:3px;">
                <b>Disclaimer -</b> This tool is strictly for Research Use only.
                It should not be used for medical purposes without consulting with doctors.
            </p>
            <p style="font-size:0.9em; text-align:center; margin-top:3px;">
                Please contact Dr. Izabela Nielsen
                (<a style="color:#800000;" href="mailto:izabela@mp.aau.dk">izabela@mp.aau.dk</a>)
                regarding any further queries.
            </p>
        </div>
    </body>
    <style>
        @media screen and (max-width: 1250px) {
            #sec-intro {
                width: 100%;
                padding-top: 0;
            }
            #sec-form {
                width: 100%;
            }
        }
    </style>
</html>
