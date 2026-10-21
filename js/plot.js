seed = cyrb128("foobar");
rand = splitmix32(seed[0]);

function cyrb128(str) {
    let h1 = 1779033703, h2 = 3144134277,
        h3 = 1013904242, h4 = 2773480762;
    for (let i = 0, k; i < str.length; i++) {
        k = str.charCodeAt(i);
        h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
        h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
        h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
        h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
    }
    h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
    h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
    h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
    h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
    h1 ^= (h2 ^ h3 ^ h4), h2 ^= h1, h3 ^= h1, h4 ^= h1;
    return [h1>>>0, h2>>>0, h3>>>0, h4>>>0];
}

function splitmix32(a) {
 return function() {
   a |= 0;
   a = a + 0x9e3779b9 | 0;
   let t = a ^ a >>> 16;
   t = Math.imul(t, 0x21f0aaad);
   t = t ^ t >>> 15;
   t = Math.imul(t, 0x735a2d97);
   return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
  }
}

function add_jitter(point)  {
    // jittered_point = point + (Math.random() - 0.5) / 3;
    var jittered_point = point + (rand() - 0.5) / 3;
    if (jittered_point < 0)
        return 0;
    return jittered_point;
}

function make_shap_plot(div_id, dataJSON, target) {
    var graphDiv = document.getElementById(div_id);
    graphDiv.innerHTML = "";

    var shap_values = JSON.parse(dataJSON).SHAP;
    var max_value = Math.max(...shap_values);

    var features = ['Age', 'WBC', 'RBC', 'Hb', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'PLT'];
    var shap_map = features.map((x, i) => [x, shap_values[i]]);
    shap_map = new Map([...shap_map].sort((a, b) => Math.abs(a[1]) - Math.abs(b[1])));

    var colors = [];
    for(var shap_value of shap_map.values())
        colors.push((shap_value < 0) ? 'blue' : 'red');


    var traces = [{
        type: 'bar',
        orientation: 'h',
        y: Array.from(shap_map.keys()),
        x: Array.from(shap_map.values()),
        // textposition: 'auto',
        marker: {
            color: colors,
            opacity: 1,
        }
    }];

    var layout = {
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        height: 400,
        uirevision: true,
        xaxis: {
            visible : true,
            zeroline: true,
            color: 'black',
            linewidth: 2,
            range: [-max_value, max_value],
            ticks: 'outside',
            ticklen: 10,
            tickwidth: 2,
            tickfont: {size: 14},
            title: {
                text: 'SHAP value for ' + target + ' regression',
                font: {size: 18}
            }
        },
        yaxis: {
            visible : true,
            zeroline: false,
            automargin: true,
            color: 'black',
            linewidth: 2,
            ticks: 'outside',
            ticklen: 10,
            tickwidth: 2,
            tickfont: {size: 14},
            title: {
                text: 'Feature',
                font: {size: 18}
            }
        },
        hoverlabel: {
            bgcolor: 'white',
            outline: {
                color: 'black'
            },
            font: {
                color: 'black',
                size: 16
            }
        },
        hovertext: {
            font: {color: 'black'}
        },
        modebar: {
            color: '#262626',
            activecolor: '#262626'
        },
        margin: {t:40}
    };

    var config = {
        showSendToCloud: false,
        displayModeBar: true,
        responsive: true,
        // displaylogo: false,
        modeBarButtonsToRemove: ['toggleSpikelines', 'zoom2d', 'select2d', 'lasso2d'],
        modeBarButtonsToAdd: [{
            name: 'Export as SVG',
            icon: {
                width: 600,
                height: 600,
                path: 'M288 32c0-17.7-14.3-32-32-32s-32 14.3-32 32V274.7l-73.4-73.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l128 128c12.5 12.5 32.8 12.5 45.3 0l128-128c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L288 274.7V32zM64 352c-35.3 0-64 28.7-64 64v32c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V416c0-35.3-28.7-64-64-64H346.5l-45.3 45.3c-25 25-65.5 25-90.5 0L165.5 352H64zm368 56a24 24 0 1 1 0 48 24 24 0 1 1 0-48z'
            },
            click: function(img) {
                Plotly.downloadImage(
                    img,
                    {
                        filename: 'hemopred_output_plot_',
                        format: 'svg'
                    }
                );
            }
        }]
    };

    Plotly.newPlot(graphDiv, traces, layout, config);
}

function make_scatter_plot(div_id, jsonData, pred_hba2, pred_hbf) {
    var graphDiv = document.getElementById(div_id);
    graphDiv.innerHTML = "";
    var data = JSON.parse(jsonData);
    var friendly_label = new Map([
        ['Beta Thalassemia Trait', 'Beta Thalassemia Trait'],
        ['Borderline high HbA2', 'Borderline HbA2'],
        ['HbD Punjab Trait', 'HbD Punjab Trait'],
        ['Compound or heterogeneous', 'HbD-Punjab/Beta Thalassemia<br> double heterozygous'],
        ['Normal', 'Normal'],
        ['Sickle Cell Trait', 'Sickle Cell Trait'],
    ]);
    var labels = [...new Set(data.label)].sort();
    var centroids = JSON.parse(data.centroids);
    var x_map = new Map();
    var y_map = new Map();
    for(var i=0; i<labels.length; ++i) {
        x_map.set(labels[i], []);
        y_map.set(labels[i], []);
    }
    for(var i=0; i<data.x.length; ++i) {
        x_map.get(data.label[i]).push(add_jitter(data.x[i]));
        y_map.get(data.label[i]).push(add_jitter(data.y[i]));
    }

    // var pt_colors = ['red', 'green', 'yellow', 'darkorange', 'blue', 'purple'];
    var pt_colors = ['#9467bd', '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#bcbd22', '#8c564b', '#e377c2', '#7f7f7f', '#17becf'];
    // var traces = [];
    var traces = [{
        mode: 'markers',
        type: 'scatter',
        x: [pred_hba2],
        y: [pred_hbf],
        name: 'Predicted HbA2 and HbF',
        opacity: 1,
        marker: {
            color: '#000000',
            symbol: 'circle',
            size: 18
        }
    }];
    for(var i=0; i<labels.length; ++i) {
        traces.push({
            mode: 'markers',
            type: 'scatter',
            x: x_map.get(labels[i]),
            y: y_map.get(labels[i]),
            name: friendly_label.get(labels[i]),
            opacity: 0.4,
            marker: {
                color: pt_colors[i],
                symbol: 'circle',
                size: 5
            }
        });
        traces.push({
            mode: 'markers',
            type: 'scatter',
            x: [centroids[i][0]],
            y: [centroids[i][1]],
            name: friendly_label.get(labels[i]) + ' (Centroid)',
            opacity: 1,
            marker: {
                color: pt_colors[i],
                symbol: 'square',
                size: 12
            }
        });
    }

    var layout = {
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        height: 500,
        scattermode: 'group',
        // scattergap: 1,
        uirevision: true,
        xaxis: {
            visible : true,
            zeroline: false,
            color: 'black',
            linewidth: 2,
            ticks: 'outside',
            ticklen: 10,
            tickwidth: 2,
            tickfont: {size: 14},
            range: [1.01, 7.8],
            title: {
                text: 'HbA2 (%)',
                font: {size: 18}
            }
        },
        yaxis: {
            visible : true,
            zeroline: false,
            color: 'black',
            linewidth: 2,
            ticks: 'outside',
            ticklen: 10,
            tickwidth: 2,
            tickfont: {size: 14},
            range: [0, 4.1],
            title: {
                text: 'HbF (%)',
                font: {size: 18}
            }
        },
        legend: {
            orientation: 'v',
            // xanchor: 'center',
            // yanchor: 'top',
            // x: -0.45,
            // y: 0.5,
            font: {size: 16},
            // maxheight: 0.5,
            // y: 1.5,
            // xanchor: 'center',
            // x: 0.5
        },
        hoverlabel: {
            bgcolor: 'white',
            outline: {
                color: 'black'
            },
            font: {
                color: 'black',
                size: 16
            }
        },
        hovertext: {
            font: {color: 'black'}
        },
        modebar: {
            color: '#262626',
            activecolor: '#262626'
        },
        margin: {t:40}
    };

    var config = {
        showSendToCloud: false,
        displayModeBar: true,
        responsive: true,
        // displaylogo: false,
        modeBarButtonsToRemove: ['toggleSpikelines', 'zoom2d', 'select2d', 'lasso2d'],
        modeBarButtonsToAdd: [{
            name: 'Export as SVG',
            icon: {
                width: 600,
                height: 600,
                path: 'M288 32c0-17.7-14.3-32-32-32s-32 14.3-32 32V274.7l-73.4-73.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l128 128c12.5 12.5 32.8 12.5 45.3 0l128-128c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L288 274.7V32zM64 352c-35.3 0-64 28.7-64 64v32c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V416c0-35.3-28.7-64-64-64H346.5l-45.3 45.3c-25 25-65.5 25-90.5 0L165.5 352H64zm368 56a24 24 0 1 1 0 48 24 24 0 1 1 0-48z'
            },
            click: function(img) {
                Plotly.downloadImage(
                    img,
                    {
                        filename: 'hemopred_output_plot_',
                        format: 'svg'
                    }
                );
            }
        }]
    };

    Plotly.newPlot(graphDiv, traces, layout, config);
}

function plot_data(div_id, pred_hba2, pred_hbf){
    var httpReq = new XMLHttpRequest();
    httpReq.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            make_scatter_plot(div_id, this.responseText, pred_hba2, pred_hbf);
        }
    };
    httpReq.open('GET', 'get_plot_data.php', true);
    httpReq.setRequestHeader("Content-type", "text/json");
    httpReq.send();
}

function plot_shap(div_id, dataJSON, target){
    var httpReq = new XMLHttpRequest();
    httpReq.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            make_shap_plot(div_id, this.responseText, target);
        }
    };
    httpReq.open('POST', 'explain.php', true);
    httpReq.setRequestHeader("Content-type", 'application/x-www-form-urlencoded');
    httpReq.send(
        'data=' + encodeURIComponent(dataJSON) +
        '&' + 'target=' + encodeURIComponent(target)
    );
}
