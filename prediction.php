<?php
    include 'validation.php';
    $valid = validateInput();
    if($valid == false) {
        echo "<center><h3>Error !!</h3></center>";
        // echo "<p>".$errMsg."</p>";
        exit("<center><i><b>Error Message:</b> ".$errMsg."</i><p>Please try again <a href=\"index.php\">here</a>.</p></center>");
    }

    $friendly_labels = array(
        'Beta Thalassemia Trait' => 'Beta Thalassemia Trait',
        'Borderline high HbA2' => 'Borderline HbA2',
        'HbD Punjab Trait' => 'HbD Punjab Trait',
        'Compound or heterogeneous' => 'HbD-Punjab/Beta Thalassemia double heterozygous',
        'Normal' => 'Normal',
        'Sickle Cell Trait' => 'Sickle Cell Trait'
    );
    $label_th_colors = array(
        "#9467bd",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#1f77b4",
        "#bcbd22",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#17becf"
    );
    $label_td_colors = array(
        "rgba(148, 103, 189, 0.4)",
        "rgba(255, 127, 14, 0.4)",
        "rgba(44, 160, 44, 0.4)",
        "rgba(214, 39, 40, 0.4)",
        "rgba(31, 119, 180, 0.4)",
        "rgba(188, 189, 34, 0.4)",
        "rgba(140, 86, 75, 0.4)",
        "rgba(227, 119, 194, 0.4)",
        "rgba(127, 127, 127, 0.4)",
        "rgba(23, 190, 207, 0.4)"
    );
    $params = array();
    $params["Age"] = $_POST["age"];
    $params["WBC"] = $_POST["wbc"];
    $params["RBC"] = $_POST["rbc"];
    $params["Hb"] = $_POST["hb"];
    $params["HCT"] = $_POST["hct"];
    $params["MCV"] = $_POST["mcv"];
    $params["MCH"] = $_POST["mch"];
    $params["MCHC"] = $_POST["mchc"];
    $params["RDW"] = $_POST["rdw"];
    $params["PLT"] = $_POST["plt"];
    $arg_json = json_encode($params);
    // echo $arg_json;

    // $command = "venv3.7/bin/python3 -m python.driver '".$arg_json."' 2>&1";
    $command = ".venv/bin/python3 -m python.webapp.driver '".$arg_json."' 2>&1";
    // echo "<pre>".$command."</pre>\n";
    exec($command, $out, $status);
    // echo implode("</br/>", $out)."<br/>";  // for checking output with python errors, if any
    // echo $out[0]."<br/>";  // for checking output only

    // echo $out[0];
    $result = json_decode($out[0]);
?>

<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction - HemoPred</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type="text/javascript" src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
        <script type = "text/javascript" src = "js/plot.js"></script>
        <style>
            #plot_div {
                width:60%;
                margin:0 20% 0 20%;
/*                 overflow: auto; */
            }
            #shap_gallery_div {
                width:60%;
                margin:0 20% 0 20%;
            }
            #shap_hba2_div {
                float:left;
                width:49%;
                margin:0;
            }
            #shap_hbf_div {
                float: right;
                width: 49%;
                margin: 0;
            }
            .output_heading {
                width:60%;
                margin:10px 20% 5px 20%;
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
                    <td class="nav"><a href="index.php" class="side_nav">Home</a></td>
                    <td class="nav"><a href="about.html" class="side_nav">About</a></td>
                    <td class="nav"><a href="help.html" class="side_nav">Help</a></td>
                    <td class="nav"><a href="team.html" class="side_nav">Team</a></td>
                </tr>
            </table>
            </center>
        </div>

        <!--<div class = "section_left"></div>-->
        <div class = "section_middle">

            <center><h2 style="margin-top:5px; margin-bottom:5px;">Inputs provided</h2></center>

            <div id="input_div" class="details">
                <table class="details">
                    <tr>
                        <th>Feature</th>
                        <th>Age</th>
                        <th>Hb</th>
                        <th>HCT</th>
                        <th>RBC count</th>
                        <th>MCV</th>
                        <th>MCH</th>
                        <th>MCHC</th>
                        <th>RDW-CV</th>
                        <th>WBC count</th>
                        <th>PLT count</th>
                    </tr>
                    <tr>
                        <td class="subheading">Value</td>
                        <td><b><?php echo $params["Age"]; ?></b></td>
                        <td><b><?php echo $params["Hb"]; ?></b></td>
                        <td><b><?php echo $params["HCT"]; ?></b></td>
                        <td><b><?php echo $params["RBC"]; ?></b></td>
                        <td><b><?php echo $params["MCV"]; ?></b></td>
                        <td><b><?php echo $params["MCH"]; ?></b></td>
                        <td><b><?php echo $params["MCHC"]; ?></b></td>
                        <td><b><?php echo $params["RDW"]; ?></b></td>
                        <td><b><?php echo $params["WBC"]; ?></b></td>
                        <td><b><?php echo $params["PLT"]; ?></b></td>
                    </tr>
                    <tr>
                        <td class="subheading">Unit</td>
                        <td>years</th>
                        <td>g/dL</td>
                        <td>%</td>
                        <td>&times; 10<sup>12</sup> /L</td>
                        <td>fL</td>
                        <td>pg</td>
                        <td>g/dL</td>
                        <td>%</td>
                        <td>&times; 10<sup>9</sup> /L</td>
                        <td>&times; 10<sup>9</sup> /L</td>
                    </tr>
                </table>
            </div>

            <center><h2 style="margin-top:10px; margin-bottom:5px;">Outputs</h2></center>

            <h4 class="output_heading">
                a. HbA2 and HbF values predicted by regression using Mixture of Experts (MoE) model.
            </h4>
            <div id="regression_output_div" class="details">
                <table class="details details_big_font">
                    <tr>
                        <th>Metric</th>
                        <th>Predicted value</th>
                        <th>Unit</th>
                        <th>95% Confidence interval</th>
                    </tr>
                    <tr>
                        <td class="subheading">Hemoglobin A2 - HbA2</td>
                        <td><b><?php echo number_format(round(floatval($result->HbA2), 3), 3, ".", ""); ?></b></td>
                        <td>(%)</td>
                        <td><b><?php echo number_format(round(floatval($result->PI_HbA2[0]), 3), 3, ".", "") . " - " . number_format(round(floatval($result->PI_HbA2[1]), 3), 3, ".", ""); ?></b></td>
                    </tr>
                    <tr>
                        <td class="subheading">Fetal Hemoglobin - HbF</td>
                        <td><b><?php echo number_format(round(floatval($result->HbF), 3), 3, ".", ""); ?></b></td>
                        <td>(%)</td>
                        <td><b><?php echo number_format(round(floatval($result->PI_HbF[0]), 3), 3, ".", "") . " - " . number_format(round(floatval($result->PI_HbF[1]), 3), 3, ".", ""); ?></b></td>
                    </tr>
                </table>
            </div>

            <h4 class="output_heading">
                b. Feature contributions for HbA2 and HbF predictions, computed by SHapley Additive exPlanation (SHAP).
            </h4>
            <div id="shap_gallery_div">
                <div id="shap_hba2_div">
                    <center><img style="height:300px;" src="resource/loading.svg" /></center>
                </div>
                <div id="shap_hbf_div">
                    <center><img style="height:300px;" src="resource/loading.svg" /></center>
                </div>
            </div>
            <div style="clear:both;"></div>

            <h4 class="output_heading">
                c. Visualization of the predicted HbA2 and HbF values.
            </h4>
            <div id="plot_div">
                <center><img style="height:300px;" src="resource/loading.svg" /></center>
            </div>

            <h4 class="output_heading">
                d. Class prediction based on affinity score computed using Nearest Centroid model.
            </h4>
            <div id="affinity_output_div" class="details">
                <table class="details details_big_font">
                    <tr>
                        <th>Label</td>
                        <th>Distance from centroid</th>
                        <th>Affinity score</th>
                        <th>Probability</th>
                    </tr>
                    <?php
                        $class_label = array_search($result->predicted_class, $result->labels);
                        for($i=0; $i<count($result->labels); ++$i) {
                            $bold_prefix = ($i == $class_label) ? "<b><u>" : "";
                            $bold_suffix = ($i == $class_label) ? "</u></b>" : "";
                            echo "<tr>";
                            echo "<th style=\"text-align:left; background-color:".$label_th_colors[$i]."\">" . $friendly_labels[$result->labels[$i]] . "</th>";
                            echo "<td style=\"background-color:".$label_td_colors[$i]."\">" . $bold_prefix . number_format(round(floatval($result->distance[$i]), 3), 3, ".", "") . $bold_suffix . "</td>";
                            echo "<td style=\"background-color:".$label_td_colors[$i]."\">" . $bold_prefix . number_format(round(floatval($result->affinity[$i]), 3), 3, ".", "") . $bold_suffix . "</td>";
                            echo "<td style=\"background-color:".$label_td_colors[$i]."\">" . $bold_prefix . number_format(round(floatval($result->proba[$i]), 3), 3, ".", "") . $bold_suffix . "</td>";
                            echo "</tr>";
                        }
                    ?>
                </table>
            </div>

            <br/><hr/>
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
    <?php echo "<script>plot_data('plot_div', ".round(floatval($result->HbA2), 3).", ".round(floatval($result->HbF), 3).");</script>"; ?>
    <?php
        echo
            "<script>
                plot_shap('shap_hba2_div', '".$arg_json."', "."'HbA2');
                plot_shap('shap_hbf_div', '".$arg_json."', "."'HbF');
            </script>";
    ?>
    <script>
         window.onresize = function() {
            Plotly.Plots.resize('plot_div');
        };
    </script>
    <style>
        @media screen and (max-width: 1600px) {
            #plot_div {
                width: 80%;
                margin: 10px 10% 10px 10%;
            }
            #shap_gallery_div {
                width: 80%;
                margin: 10px 10% 10px 10%;
            }
        }
        @media screen and (max-width: 1250px) {
            #input_div {
                width: 80%;
                margin: 0 10% 0 10%;
            }
            #regression_output_div {
                width: 80%;
                margin: 0 10% 0 10%;
            }
            #plot_div {
                width: 96%;
                margin: 10px 2% 10px 2%;
            }
            #shap_gallery_div {
                width: 96%;
                margin: 10px 2% 10px 2%;
            }
            #affinity_output_div {
                width: 80%;
                margin: 0 10% 0 10%;
            }
            .output_heading {
                width: 80%;
                margin: 0 10% 0 10%;
            }
        }
        @media screen and (max-width: 1000px) {
            #input_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #regression_output_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #plot_div {
                width: 98%;
                margin: 10px 1% 10px 1%;
            }
            #shap_gallery_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #shap_hba2_div {
                width: 100%;
                margin: 0;
            }
            #shap_hbf_div {
                width: 100%;
                margin: 0;
            }
            #affinity_output_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            .output_heading {
                width: 98%;
                margin: 0 1% 0 1%;
            }
        }
        @media print {
            #input_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #regression_output_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #plot_div {
                width: 98%;
                margin: 10px 1% 10px 1%;
            }
            #shap_gallery_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            #affinity_output_div {
                width: 98%;
                margin: 0 1% 0 1%;
            }
            .output_heading {
                width: 98%;
                margin: 0 1% 0 1%;
            }
        }
    </style>
</html>
