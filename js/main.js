var demoData = [
    [26, 10.3, 5.07, 10.2, 32.4, 63.8, 20.1, 31.5, 16.3, 222],
    [30, 5.9, 3.28, 6.8, 23.7, 72.3, 20.6, 28.5, 22.6, 164],
    [33, 8.0, 5.65, 14.7, 44.9, 79.4, 26.0, 32.7, 15.2, 174],
];
function addDemoData(index) {
    var data = demoData[index];
    for(var i=0; i<data.length; ++i)
        document.getElementById('v'+(i+1)).value = data[i];
}
