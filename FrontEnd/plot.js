// frontend/plot.js
// expects response format from backend /train
function renderModelCanvasPlot(data) {
  const train = data.train;
  const test = data.test;
  const grid = data.grid;
  const classes = data.classes;

  // scatter traces for each class (train)
  const traces = [];
  const unique = [...new Set(train.labels.concat(test.labels))].sort();

  // colors are left to Plotly default; that meets the "no fixed color" guidance
  unique.forEach(cls => {
    // train points of this class
    const t_x = [];
    const t_y = [];
    train.labels.forEach((lab, i) => {
      if (String(lab) === String(cls)) {
        t_x.push(train.x[i]);
        t_y.push(train.y[i]);
      }
    });
    traces.push({
      x: t_x,
      y: t_y,
      mode: 'markers',
      name: `train: ${cls}`,
      marker: { size: 8, symbol: 'circle' }
    });
  });

  // test points overlay (hollow markers)
  const test_trace = {
    x: test.x,
    y: test.y,
    mode: 'markers',
    name: 'test (predictions)',
    marker: {
      size: 10,
      symbol: 'diamond',
      opacity: 0.9,
      line: { width: 1 }
    },
    text: test.preds.map((p, i) => `pred:${p} actual:${test.labels[i]}`)
  };
  traces.push(test_trace);

  // grid as density contour: we need to reshape
  const nx = grid.nx;
  const ny = grid.ny;
  // rebuild grid matrix
  const xx = grid.xx;
  const yy = grid.yy;
  const preds = grid.preds;

  // convert list of flattened points back into 2D shape suitable for z
  // We know grid points were produced by meshgrid with shape (ny, nx)
  // z should be of shape [ny][nx]
  const z = [];
  for (let r = 0; r < grid.ny; r++) {
    const row = [];
    for (let c = 0; c < grid.nx; c++) {
      const idx = r * grid.nx + c;
      row.push(preds[idx]);
    }
    z.push(row);
  }

  // To overlay boundary, draw a heatmap or contour with low opacity
  const xi = [];
  for (let c = 0; c < grid.nx; c++) xi.push(grid.x_min + c * ((grid.x_max - grid.x_min) / (grid.nx - 1)));
  const yi = [];
  for (let r = 0; r < grid.ny; r++) yi.push(grid.y_min + r * ((grid.y_max - grid.y_min) / (grid.ny - 1)));

  const boundary = {
    x: xi,
    y: yi,
    z: z,
    type: 'contour',
    colorscale: 'RdBu',
    opacity: 0.4,
    showscale: false,
    contours: { coloring: 'heatmap', showlines: false }
  };

  // Put boundary beneath points by placing it first
  const plotData = [boundary, ...traces];

  const layout = {
    title: `${data.dataset} — ${data.algo} (acc ${data.accuracy})`,
    margin: { l: 40, r: 20, t: 60, b: 40 },
    xaxis: { title: 'PC1' },
    yaxis: { title: 'PC2' },
    legend: { orientation: "h", x: 0, y: -0.12 }
  };

  Plotly.react('plot', plotData, layout, {responsive: true});
}
