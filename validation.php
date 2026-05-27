<?php
$errMsg = "";
    function checkExistence() {
        global $errMsg;
        if (!array_key_exists("age", $_POST)) {
            $errMsg = "Age value does not exist";
            return false;
        }
        if (!array_key_exists("wbc", $_POST)) {
            $errMsg = "WBC value does not exist";
            return false;
        }
        if (!array_key_exists("rbc", $_POST)) {
            $errMsg = "RBC value does not exist";
            return false;
        }
        if (!array_key_exists("hb", $_POST)) {
            $errMsg = "Hb value does not exist";
            return false;
        }
        if (!array_key_exists("hct", $_POST)) {
            $errMsg = "HCT value does not exist";
            return false;
        }
        if (!array_key_exists("mcv", $_POST)) {
            $errMsg = "MCV value does not exist";
            return false;
        }
        if (!array_key_exists("mch", $_POST)) {
            $errMsg = "MCH value does not exist";
            return false;
        }
        if (!array_key_exists("mchc", $_POST)) {
            $errMsg = "MCHC value does not exist";
            return false;
        }
        if (!array_key_exists("rdw", $_POST)) {
            $errMsg = "RDW value does not exist";
            return false;
        }
        if (!array_key_exists("plt", $_POST)) {
            $errMsg = "Platelet count value does not exist";
            return false;
        }
        return true;
    }

    function checkRanges() {
        global $errMsg;
        if(!is_numeric($_POST["age"]) or intval($_POST["age"])<18 or intval($_POST["age"])>90) {
            $errMsg = "Invalid Age value";
            return false;
        }
        if(!is_numeric($_POST["wbc"]) or floatval($_POST["wbc"])<0.2 or floatval($_POST["wbc"])>150.0) {
            $errMsg = "Invalid WBC value";
            return false;
        }
        if(!is_numeric($_POST["rbc"]) or floatval($_POST["rbc"])<0.50 or floatval($_POST["rbc"])>9.00) {
            $errMsg = "Invalid RBC value";
            return false;
        }
        if(!is_numeric($_POST["hb"]) or floatval($_POST["hb"])<1.0 or floatval($_POST["hb"])>22.0) {
            $errMsg = "Invalid Hb value";
            return false;
        }
        if(!is_numeric($_POST["hct"]) or floatval($_POST["hct"])<10.0 or floatval($_POST["hct"])>70.0) {
            $errMsg = "Invalid HCT value";
            return false;
        }
        if(!is_numeric($_POST["mcv"]) or floatval($_POST["mcv"])<35.0 or floatval($_POST["mcv"])>170.0) {
            $errMsg = "Invalid MCV value";
            return false;
        }
        if(!is_numeric($_POST["mch"]) or floatval($_POST["mch"])<10.0 or floatval($_POST["mch"])>50.0) {
            $errMsg = "Invalid MCH value";
            return false;
        }
        if(!is_numeric($_POST["mchc"]) or floatval($_POST["mchc"])<20.0 or floatval($_POST["mchc"])>50.0) {
            $errMsg = "Invalid MCHC value";
            return false;
        }
        if(!is_numeric($_POST["rdw"]) or floatval($_POST["rdw"])<10.0 or floatval($_POST["rdw"])>50.0) {
            $errMsg = "Invalid RDW-CV value";
            return false;
        }
        if(!is_numeric($_POST["plt"]) or intval($_POST["plt"])<1 or intval($_POST["plt"])>1500) {
            $errMsg = "Invalid PLT value";
            return false;
        }
        return true;
    }

    function validateInput() {
        if(!checkExistence())
            return false;
        if(!checkRanges())
            return false;
        return true;
    }
?>
