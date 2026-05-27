<?php
    $arg_json = urldecode($_POST["data"]);
    $target = urldecode($_POST["target"]);
    // echo $arg_json;

    $command = ".venv/bin/python3 -m python.webapp.driver_explain '".$arg_json."' '".$target."'  2>&1";
    // echo "<pre>".$command."</pre>\n";
    exec($command, $out, $status);
    // echo implode("</br/>", $out)."<br/>";  // for checking output with python errors, if any
    // echo $out[0]."<br/>";  // for checking output only

    echo $out[0];
    // $result = json_decode($out[0]);
    // echo $result->SHAP;
?>
