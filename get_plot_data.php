<?php
    $x = array();
    $y = array();
    $label = array();
    $f = fopen("output/affinity/ac_pred_euclidean_inverse.tsv", "r");
    fgetcsv($f, 0, "\t");
    while (($row = fgetcsv($f, 0, "\t")) !== false) {
        array_push($x, floatval($row[1]));
        array_push($y, floatval($row[2]));
        array_push($label, $row[3]);
    }
    fclose($f);
    $f = fopen("output/affinity/ac_pred_euclidean_inverse.tsv", "r");
    fgetcsv($f, 0, "\t");
    $centroid_string = fgetcsv($f, 0, "\t")[7];
    fclose($f);


    echo json_encode(array("x" => $x, "y" => $y, "label" => $label, "centroids" => $centroid_string));
?>
