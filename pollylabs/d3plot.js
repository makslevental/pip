const POINT_SIZE = 4;

function pointMouseEnter(ref) {
  d3.select(ref).attr("r", 2 * POINT_SIZE);
}

function pointMouseLeave(ref) {
  d3.select(ref).attr("r", POINT_SIZE);
}

function pointsDimRange_native(points, dim) {
  let min = points[0][dim];
  let max = points[0][dim];
  for (let p of points) {
    if (p[dim] < min)
      min = p[dim];
    if (p[dim] > max)
      max = p[dim];
  }
  return [min, max];
}

function pointsDimRange(points, dim) {
  return d3.extent(points, p => p[dim]);
}

function steps(begin, end, step) {
  s = [];
  for (let i = begin; i < end; i += step)
    s.push(i | 0);
  return s;
}

// Points = array of 2D arrays
function populateSvg(parentSelector, points) {
  let width = 420;
  let height = 420;
  let padding = 10;
  let axisPadding = 20;

  let xRange = pointsDimRange(points, 0);
  let yRange = pointsDimRange(points, 1);
  let xScale = d3.scaleLinear()
    .domain(xRange)
    .range([0 + padding + axisPadding, width - padding]);
  let yScale = d3.scaleLinear()
    .domain(yRange)
    .range([height - padding - axisPadding, 0 + padding]);
    //.range([0 + padding, height - padding - axisPadding]); // for inversion

  let xAxis = d3.axisBottom(xScale)
    .ticks(6);
  let yAxis = d3.axisLeft(yScale)
    .ticks(6);

  let parentSelection = d3.select(parentSelector)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  parentSelection.append("g")
    .attr("class", "axis x")
    .attr("transform", "translate(0, " + (height - axisPadding) + ")")
    .call(xAxis);
  parentSelection.append("g")
    .attr("class", "axis y")
    .attr("transform", "translate(" + axisPadding + ", 0)")
    .call(yAxis);

  parentSelection.append("rect")
    .attr("class", "frame")
    .attr("x", axisPadding)
    .attr("y", padding / 2)
    .attr("width", width - axisPadding - padding/2)
    .attr("height", height - axisPadding - padding/2);

  parentSelection.selectAll("circle")
    .data(points)
    .enter()
    .append("circle")
    .attr("class", "isl-point")
    .attr("r", POINT_SIZE)
    .attr("cx", d => xScale(d[0]))
    .attr("cy", d => yScale(d[1]))
    .attr("onmouseenter", "pointMouseEnter(this)")
    .attr("onmouseleave", "pointMouseLeave(this)");
};

function highlightPointTicks(ref, xTicklineId, yTicklineId,
    xTickId, yTickId, turnOn) {
  d3.select(ref)
    .attr("r", (turnOn ? 1.5 : 1) * POINT_SIZE);
  d3.select(xTicklineId)
    .style("stroke", turnOn ? "#333" : "#eee");
  d3.select(yTicklineId)
    .style("stroke", turnOn ? "#333" : "#eee");
  d3.select(xTickId)
    .style("font-weight", turnOn ? "bold" : "normal");
  d3.select(yTickId)
    .style("font-weight", turnOn ? "bold" : "normal");
}

function createSvg(parentSelector) {
  const width = 420;
  const height = 420;
  const legendMargin = 60;

  let svg = d3.select(parentSelector)
    .append("svg")
    .attr("width", width + legendMargin)
    .attr("height", height);
  
  return svg;
}

function _plotUnionSet_getScales(width, height, padding,
    axisPadding, xRange, yRange) {

  let xScale = d3.scaleLinear()
    .domain(xRange)
    .range([0 + padding + axisPadding, width - padding]);
  let yScale = d3.scaleLinear()
    .domain(yRange)
    .range([height - padding - axisPadding, 0 + padding]);

  return [xScale, yScale];
}

function _plotUnionSet_plotFrame(svg, graphid, padding,
    axisPadding, xScale, yScale) {
  const xTicklineIdPrefix = graphid + "_tickline_x";
  const yTicklineIdPrefix = graphid + "_tickline_y";
  const xTickIdPrefix = graphid + "_tick_x";
  const yTickIdPrefix = graphid + "_tick_y";

  let xRange = xScale.domain();
  let yRange = yScale.domain();

  let xTickLines = svg.append("g")
    .selectAll(".tickline")
    .data(steps(xRange[0], xRange[1] + 1, 1))
    .enter()
    .append("g");
  xTickLines.append("line")
    .attr("class", "tickline x")
    .attr("x1", d => xScale(d))
    .attr("y1", yScale(yRange[0]) + padding/2)
    .attr("x2", d => xScale(d))
    .attr("y2", yScale(yRange[1]) - padding/2)
    .attr("id", d => xTicklineIdPrefix + d);
  xTickLines.append("text")
    .attr("class", "ticktext x")
    .attr("x", d => xScale(d))
    .attr("y", yScale(yRange[0]) + axisPadding)
    .attr("id", d => xTickIdPrefix + d)
    .style("text-anchor", "middle")
    .text(d => "" + (d | 0));

  let yTickLines = svg.append("g")
    .selectAll(".tickline")
    .data(steps(yRange[0], yRange[1] + 1, 1))
    .enter()
    .append("g");
  yTickLines.append("line")
    .attr("class", "tickline")
    .attr("x1", xScale(xRange[0]) - padding/2)
    .attr("y1", d => yScale(d))
    .attr("x2", xScale(xRange[1]) + padding/2)
    .attr("y2", d => yScale(d))
    .attr("id", d => yTicklineIdPrefix + d);
  yTickLines.append("text")
    .attr("class", "ticktext y")
    .attr("x", xScale(xRange[0]) - axisPadding)
    .attr("y", d => yScale(d))
    .attr("dy", "0.3em")
    .attr("id", d => yTickIdPrefix + d)
    .style("text-anchor", "begin")
    .text(d => "" + (d | 0));
}

function _plotUnionSet_plotTiles(svg, uset, xScale, yScale, nameScale) {
  svg.selectAll(".tiles")
    .data(uset)
    .enter()
    .append("g")
    .each(function (set) {
      let tiles = set["tiles"];
      d3.select(this)
        .selectAll(".tile")
        .data(tiles)
        .enter()
        .append("path")
        .attr("class", "tile")
        .attr("d", function (tile) {
          let p = d3.path();
          p.moveTo(xScale(tile[0][0]), yScale(tile[0][1]));
          for (let i = 0; i < tile.length; ++i) {
            p.lineTo(xScale(tile[i][0]), yScale(tile[i][1]));
          }
          p.closePath();  
          return p.toString();
        })
        .style("fill", nameScale(set["name"]));
    });
}

function _plotUnionSet_plotPoints(svg, uset, graphid, xScale, yScale, nameScale) {
  const xTicklineIdPrefix = graphid + "_tickline_x";
  const yTicklineIdPrefix = graphid + "_tickline_y";
  const xTickIdPrefix = graphid + "_tick_x";
  const yTickIdPrefix = graphid + "_tick_y";

  svg.selectAll(".statement")
    .data(uset)
    .enter()
    .append("g")
    .attr("class", "statement")
    .each(function (set) {
      d3.select(this)
        .selectAll("circle")
        .data(set["points"])
        .enter()
        .append("circle")
        .attr("class", "point")
        .attr("r", POINT_SIZE)
        .attr("cx", d => xScale(d[0]))
        .attr("cy", d => yScale(d[1]))
        .style("fill", nameScale(set["name"]))
        .attr("onmouseenter", 
              d => "highlightPointTicks(this,\"#"
                  + xTicklineIdPrefix + d[0] + "\", \"#"
                  + yTicklineIdPrefix + d[1] + "\", \"#"
                  + xTickIdPrefix + d[0] + "\", \"#"
                  + yTickIdPrefix + d[1] + "\", true)")
        .attr("onmouseleave",
              d => "highlightPointTicks(this,\"#"
                  + xTicklineIdPrefix + d[0] + "\", \"#"
                  + yTicklineIdPrefix + d[1] + "\", \"#"
                  + xTickIdPrefix + d[0] + "\", \"#"
                  + yTickIdPrefix + d[1] + "\", false)");
    });
}

function _plotUnionSet_plotLegend(svg, names, width, height, padding, nameScale) {
  let legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", "translate(" + (width + padding) + "," + 0.4 * height + ")");

  let legendElements = legend.selectAll("circle")
    .data(names)
    .enter()
    .append("g");

  legendElements.append("circle")
    .attr("r", POINT_SIZE)
    .attr("cx", 0)
    .attr("cy", (d, i) => i * 15)
    .style("fill", d => nameScale(d));
  legendElements.append("text")
    .attr("x", POINT_SIZE + 5)
    .attr("y", (d, i) => i * 15)
    .attr("dy", "0.3em")
    .style("text-anchor", "begin")
    .text(d => d);
}

function _plotUnionSet_inRange(svg, uset, graphid, xRange, yRange) {
  const width = (svg.attr("width") | 0) - 60;
  const height = (svg.attr("height") | 0);
  const padding = 10;
  const axisPadding = 20;

  let names = uset.map(d => d["name"]);
  let nameScale = d3.scaleOrdinal(d3.schemeCategory10);
  let [xScale, yScale] = _plotUnionSet_getScales(width, height, padding,
    axisPadding, xRange, yRange);

  _plotUnionSet_plotFrame(svg, graphid, padding, axisPadding, xScale, yScale);
  _plotUnionSet_plotTiles(svg, uset, xScale, yScale, nameScale);
  _plotUnionSet_plotPoints(svg, uset, graphid, xScale, yScale, nameScale);
  _plotUnionSet_plotLegend(svg, names, width, height, padding, nameScale);
}

function extractRange(obj, key, idx) {
  const extents = obj.map(d => pointsDimRange(d[key], idx));
  return d3.extent(d3.merge(extents));
}

// uset = array of objects {"name": string identifier, "points": array of 2-element arrays with coordinates, "tiles": array of arrays of 2-element arrays with points}
function plotUnionSetCombined_(svg, uset, graphid) {
  let xRange = extractRange(uset, "points", 0);
  let yRange = extractRange(uset, "points", 1);

  _plotUnionSet_inRange(svg, uset, graphid, xRange, yRange);
}

function plotUnionSetCombined(parentSelector, uset, graphid) {
  plotUnionSetCombined_(createSvg(parentSelector), uset, graphid);
}

function crossName(f,t) {
  return f + "->" + t;
}

function crossNames(umap) {
  result = [];
  for (let pair of umap.map(d => crossName(d["from"], d["to"]))) {
    if (result.indexOf(pair) == -1)
      result.push(pair);
  }
  return result;
}

function unionNames(umap) {
  let fromNames = umap.map(d => d["from"]);
  let toNames = umap.map(d => d["to"]);
  result = [];
  for (let t of fromNames.concat(toNames)) {
    if (result.indexOf(t) == -1)
      result.push(t);
  }
  return result;
}

function combinePoints(part1, part2) {
  let strs = part1.map(x => x.toString());
  let result = part1;
  for (let p of part2) {
    if (strs.indexOf(p.toString()) == -1)
      result.push(p);
  }
  return result;
}

function projectUnionMap(umap, allNames) {
  result = [];
  for (let name of allNames) {
    current = {"name": name, "points": [], "tiles": []};
    for (let m of umap) {
      if (m["from"] == name) {
        let pts = m["points"].map(d => d.slice(0,2));
        current["points"] = combinePoints(current["points"], pts);
      }
      if (m["to"] == name) {
        let pts = m["points"].map(d => d.slice(2,4));
        current["points"] = combinePoints(current["points"], pts);
      }
    }
    result.push(current);
  }
  return result;
}

// create svg marks for different-colored arrow heads
function _plotUnionMap_addArrowHeadDefs(svg, depNames, depScale, graphid) {
  svg.append("defs")
    .selectAll("marker")
    .data(depNames)
    .enter()
    .append("marker")
    .attr("id", d => graphid + "_arrowhead_" + d.replace(">",""))
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 9)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .style("fill", d => depScale(d))
    .append("path")
    .attr("d", "M0,-5L10,0L0,5");
}

// umap = array of objects {"from": string identifier, "to": string identifier, "points": array of 2-element arrays [x1,y1,x2,y2], first to belong to from}
// "Closed" in the name means that the domain and range of the map considered to be in the same space, therefore only one scatterplot is dispayed
function plotUnionMapClosed_(svg, umap, graphid) {
  const width = (svg.attr("width") | 0) - 60;
  const height = (svg.attr("height") | 0);
  const padding = 10;
  const axisPadding = 20;

  let allNames = unionNames(umap);
  let depNames = crossNames(umap);

  let nameScale = d3.scaleOrdinal(d3.schemeCategory10);
  let depScale = d3.scaleOrdinal(d3.schemeCategory10);

  let x1Range = extractRange(umap, "points", 0);
  let y1Range = extractRange(umap, "points", 1);
  let x2Range = extractRange(umap, "points", 2);
  let y2Range = extractRange(umap, "points", 3);

  let xRange = d3.extent(d3.merge([x1Range, x2Range]));
  let yRange = d3.extent(d3.merge([y1Range, y2Range]));
  let [xScale, yScale] = _plotUnionSet_getScales(width, height, padding,
    axisPadding,xRange, yRange);

  _plotUnionMap_addArrowHeadDefs(svg, depNames, depScale, graphid);

  _plotUnionSet_plotFrame(svg, graphid, padding, axisPadding, xScale, yScale);
  _plotUnionSet_plotPoints(svg, projectUnionMap(umap, allNames),
      graphid, xScale, yScale, nameScale);

  svg.selectAll(".dep")
    .data(umap)
    .enter()
    .append("g")
    .attr("class", "dep")
    .each(function (mp) {
      // normal lines
      d3.select(this)
        .selectAll("line")
        .data(mp["points"])
        .enter()
        .filter(d => ((d[0] != d[2]) || (d[1] != d[3])))
        .append("line")
        .attr("class", "depline")
        .attr("x1", d => xScale(d[0]))
        .attr("y1", d => yScale(d[1]))
        .attr("x2", d => xScale(d[2]))
        .attr("y2", d => yScale(d[3]))
        .style("stroke", depScale(crossName(mp["from"], mp["to"])))
        .style("marker-end", "url(#" + graphid + "_arrowhead_" + mp["from"] + "-" + mp["to"] + ")");
      // self-lines (make circles)
      d3.select(this)
        .selectAll("circle")
        .data(mp["points"])
        .enter()
        .filter(d => (d[0] == d[2] && d[1] == d[3]))
        .append("circle")
        .attr("class", "depline")
        .attr("cx", d => xScale(d[0]) + POINT_SIZE)
        .attr("cy", d => yScale(d[1]) - POINT_SIZE)
        .attr("r", POINT_SIZE*2)
        .style("fill", "none")
        .style("stroke", depScale(crossName(mp["from"], mp["to"])));
    });

  _plotUnionSet_plotLegend(svg, allNames, width, height/2, padding, nameScale);
  _plotUnionSet_plotLegend(svg, depNames, width, height*2, padding, depScale);
}

function plotUnionMapClosed(parentSelector, umap, graphid) {
  plotUnionMapClosed_(createSvg(parentSelector), umap, graphid);
}
