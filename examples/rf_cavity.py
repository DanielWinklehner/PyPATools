import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json


# Read the Excel file
def load_collaboration_data(filename='collaboration_data.xlsx'):
    """Load collaboration matrix from Excel"""
    df = pd.read_excel(filename, index_col=0)
    print(f"📋 Loaded dataframe shape: {df.shape}")
    print(f"📋 Index: {df.index.tolist()}")
    print(f"📋 Columns: {df.columns.tolist()}")

    # Extract location columns if they exist
    location_cols = ['PI1_Location', 'PI2_Location']
    locations_df = None

    if all(col in df.columns for col in location_cols):
        locations_df = df[location_cols].copy()
        # Remove location columns from main dataframe (won't be in table)
        df = df.drop(columns=location_cols)
        print(f"📍 Location data extracted for {len(locations_df)} members")

    return df, locations_df


# Calculate positions for nodes
def calculate_positions():
    """Calculate (x, y) positions for all members"""
    positions = {}
    radius = 0.8

    # A members: Left arc - MIRROR of B (A01-A08)
    a_members = [f'A{i:02d}' for i in range(1, 9)]
    n_a = len(a_members)
    angles_a = np.linspace(np.pi * 0.75, np.pi * 1.25, n_a)

    for i, member in enumerate(a_members):
        positions[member] = {
            'x': radius * np.cos(angles_a[i]),
            'y': radius * np.sin(angles_a[i])
        }

    # B members: Right arc (B01-B09)
    b_members = [f'B{i:02d}' for i in range(1, 10)]
    n_b = len(b_members)
    angles_b = np.linspace(np.pi * 0.25, -np.pi * 0.25, n_b)

    for i, member in enumerate(b_members):
        positions[member] = {
            'x': radius * np.cos(angles_b[i]),
            'y': radius * np.sin(angles_b[i])
        }

    # Z members: Center vertical (Z01-Z02)
    z_members = ['Z01', 'Z02']
    z_spacing = 0.3
    for i, member in enumerate(z_members):
        positions[member] = {
            'x': 0.0,
            'y': z_spacing - (i * z_spacing * 2)
        }

    return positions


# Generate HTML table from dataframe
def generate_html_table(df):
    """Generate HTML table with data attributes for highlighting"""
    members = df.index.tolist()

    table_html = '<table id="dataTable">\n'

    # Header row
    table_html += '  <thead>\n    <tr>\n      <th class="corner-cell"></th>\n'
    for col in df.columns:
        table_html += f'      <th class="col-header" data-member="{col}">{col}</th>\n'
    table_html += '    </tr>\n  </thead>\n'

    # Data rows
    table_html += '  <tbody>\n'
    for row_member in members:
        table_html += f'    <tr data-member="{row_member}">\n'
        table_html += f'      <th class="row-header" data-member="{row_member}">{row_member}</th>\n'
        for col_member in df.columns:
            value = df.loc[row_member, col_member]
            display_value = '' if pd.isna(value) or value == '' else str(value)
            table_html += f'      <td data-row="{row_member}" data-col="{col_member}">{display_value}</td>\n'
        table_html += '    </tr>\n'
    table_html += '  </tbody>\n'
    table_html += '</table>\n'

    return table_html


# Create interactive plot with table
def create_collaboration_plot(df, locations_df, positions, output_file='collaboration_network.html'):
    """Create interactive collaboration network with synchronized table"""

    members = df.index.tolist()

    # Build connection data with BOTH incoming and outgoing connections
    connections = {}
    for member in members:
        connections[member] = {}

        # Check all other members
        for other_member in members:
            if member != other_member:
                outgoing_value = None
                incoming_value = None

                # Check outgoing (member -> other_member)
                try:
                    val = df.loc[member, other_member]
                    if pd.notna(val) and val != '':
                        outgoing_value = str(val)
                except KeyError:
                    pass

                # Check incoming (other_member -> member)
                try:
                    val = df.loc[other_member, member]
                    if pd.notna(val) and val != '':
                        incoming_value = str(val)
                except KeyError:
                    pass

                # Store if there's any connection
                if outgoing_value or incoming_value:
                    connections[member][other_member] = {
                        'outgoing': outgoing_value,
                        'incoming': incoming_value
                    }

    # Prepare node data with locations
    nodes_data = []
    member_locations = {}

    for member in members:
        if member in positions:
            node_info = {
                'id': member,
                'x': positions[member]['x'],
                'y': positions[member]['y'],
                'type': member[0]
            }

            # Add location data if available
            if locations_df is not None and member in locations_df.index:
                locs = []
                pi1_loc = locations_df.loc[member, 'PI1_Location']
                pi2_loc = locations_df.loc[member, 'PI2_Location']

                if pd.notna(pi1_loc) and pi1_loc != '':
                    locs.append(str(pi1_loc))
                if pd.notna(pi2_loc) and pi2_loc != '':
                    locs.append(str(pi2_loc))

                node_info['locations'] = locs
                member_locations[member] = locs
            else:
                node_info['locations'] = []
                member_locations[member] = []

            nodes_data.append(node_info)

    # Generate table HTML
    table_html = generate_html_table(df)

    # Create HTML with Plotly and custom JavaScript
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}

        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }}

        .container {{
            display: flex;
            height: 100vh;
            width: 100vw;
        }}

        .plot-container {{
            flex: 0 0 30%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            min-width: 200px;
        }}

        .resizer {{
            flex: 0 0 5px;
            background-color: #ddd;
            cursor: col-resize;
            position: relative;
            transition: background-color 0.2s;
        }}

        .resizer:hover {{
            background-color: #007bff;
        }}

        .resizer::after {{
            content: '⋮';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #666;
            font-size: 16px;
        }}

        .table-container {{
            flex: 1;
            padding: 20px;
            display: flex;
            flex-direction: column;
            min-width: 300px;
        }}

        h2 {{
            margin: 0 0 10px 0;
            font-size: 18px;
            text-align: center;
            color: #333;
            flex-shrink: 0;
        }}

        #plotDiv {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: calc(100vh - 280px);
            width: 100%;
        }}

        .table-header {{
            display: flex;
            justify-content: flex-start;
            align-items: center;
            margin-bottom: 10px;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .table-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .control-btn {{
            padding: 6px 12px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.2s;
            white-space: nowrap;
        }}

        .control-btn:hover {{
            background-color: #0056b3;
        }}

        .zoom-controls {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}

        .zoom-btn {{
            padding: 4px 8px;
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }}

        .zoom-btn:hover {{
            background-color: #5a6268;
        }}

        .zoom-level {{
            font-size: 12px;
            color: #666;
            min-width: 40px;
            text-align: center;
        }}

        .table-wrapper {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            flex: 1;
            overflow: auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }}

        .table-content {{
            transform-origin: top left;
            transition: transform 0.2s;
            min-height: 100%;
            display: inline-block;
        }}

        #dataTable {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}

        #dataTable th,
        #dataTable td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            transition: background-color 0.2s;
        }}

        #dataTable th {{
            background-color: #f8f9fa;
            font-weight: bold;
            position: sticky;
            z-index: 10;
        }}

        #dataTable thead th {{
            top: 0;
        }}

        #dataTable .row-header {{
            left: 0;
            background-color: #f8f9fa;
            font-weight: bold;
        }}

        #dataTable .corner-cell {{
            left: 0;
            top: 0;
            z-index: 20;
            background-color: #e9ecef;
        }}

        /* Highlighting styles */
        #dataTable th.highlighted,
        #dataTable td.highlighted {{
            background-color: #fff3cd !important;
            font-weight: bold;
        }}

        .info {{
            text-align: center;
            color: #666;
            margin: 10px 0;
            font-size: 14px;
            flex-shrink: 0;
        }}

        .pin-notice {{
            text-align: center;
            color: #007bff;
            font-weight: bold;
            font-size: 12px;
            min-height: 18px;
            margin: 5px 0;
        }}

        .legend {{
            text-align: left;
            margin: 15px 10px 10px 10px;
            flex-shrink: 0;
            font-size: 11px;
            line-height: 1.8;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}

        .legend-color {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            flex-shrink: 0;
            border: 2px solid white;
        }}

        .legend-arrow {{
            display: inline-flex;
            align-items: center;
            margin-right: 8px;
            flex-shrink: 0;
        }}

        .legend-arrow svg {{
            width: 30px;
            height: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left side: Plot -->
        <div class="plot-container" id="plotContainer">
            <h2>Collaboration Network</h2>
            <div id="plotDiv"></div>
            <div class="info">💡 Hover: node or location box | Click: pin/unpin</div>
            <div class="pin-notice" id="pinNotice"></div>
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color" style="background-color: rgb(120, 162, 133);"></span>
                    <span><b>A</b> - Molecular mechanisms of immunothrombosis and thromboinflammation</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: rgb(247, 156, 100);"></span>
                    <span><b>B</b> - Targeting immunothrombosis and thromboinflammation in disease</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: rgb(127, 196, 224);"></span>
                    <span><b>Z</b> - Central service projects and integrated research training group</span>
                </div>
                <div class="legend-item">
                    <span class="legend-arrow">
                        <svg viewBox="0 0 30 16">
                            <defs>
                                <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto">
                                    <polygon points="0 0, 5 3, 0 6" fill="#666" />
                                </marker>
                            </defs>
                            <line x1="2" y1="8" x2="28" y2="8" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />
                        </svg>
                    </span>
                    <span><b>provides</b></span>
                </div>
            </div>
        </div>

        <!-- Resizer -->
        <div class="resizer" id="resizer"></div>

        <!-- Right side: Table -->
        <div class="table-container" id="tableContainer">
            <div class="table-header">
                <h2>Collaboration Data Matrix</h2>
                <div class="table-controls">
                    <div class="zoom-controls">
                        <button class="zoom-btn" id="zoomOut" title="Zoom Out">−</button>
                        <span class="zoom-level" id="zoomLevel">100%</span>
                        <button class="zoom-btn" id="zoomIn" title="Zoom In">+</button>
                    </div>
                    <button class="control-btn" id="downloadCSV">📥 CSV</button>
                    <button class="control-btn" id="resetTable">🔄 Reset</button>
                </div>
            </div>
            <div class="table-wrapper" id="tableWrapper">
                <div class="table-content" id="tableContent">
                    {table_html}
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data
        var nodes = {json.dumps(nodes_data)};
        var connections = {json.dumps(connections)};
        var memberLocations = {json.dumps(member_locations)};

        // Updated color scheme
        var colors = {{
            'A': 'rgb(120, 162, 133)',
            'B': 'rgb(247, 156, 100)',
            'Z': 'rgb(127, 196, 224)'
        }};

        // Location colors
        var locationColors = {{
            'Berlin': 'rgb(21, 80, 149)',
            'Mainz': 'rgb(193, 11, 37)',
            'Munich': 'rgb(0, 137, 59)',
            'Würzburg': 'rgb(0, 74, 145)'
        }};

        var locationNames = ['Berlin', 'Mainz', 'Munich', 'Würzburg'];

        // Build location to members mapping
        var locationToMembers = {{}};
        locationNames.forEach(loc => {{
            locationToMembers[loc] = [];
        }});

        Object.keys(memberLocations).forEach(member => {{
            var locs = memberLocations[member];
            locs.forEach(loc => {{
                if (locationToMembers[loc]) {{
                    locationToMembers[loc].push(member);
                }}
            }});
        }});

        // Resizer functionality
        var resizer = document.getElementById('resizer');
        var plotContainer = document.getElementById('plotContainer');
        var tableContainer = document.getElementById('tableContainer');
        var container = document.querySelector('.container');

        var isResizing = false;

        resizer.addEventListener('mousedown', function(e) {{
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        }});

        document.addEventListener('mousemove', function(e) {{
            if (!isResizing) return;

            var containerRect = container.getBoundingClientRect();
            var newPlotWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

            newPlotWidth = Math.max(15, Math.min(70, newPlotWidth));

            plotContainer.style.flex = '0 0 ' + newPlotWidth + '%';

            Plotly.Plots.resize('plotDiv');
        }});

        document.addEventListener('mouseup', function() {{
            if (isResizing) {{
                isResizing = false;
                document.body.style.cursor = 'default';
            }}
        }});

        // Create grey lines only for connected nodes
        var greyLines = [];
        for (var i = 0; i < nodes.length; i++) {{
            for (var j = i + 1; j < nodes.length; j++) {{
                var node1 = nodes[i].id;
                var node2 = nodes[j].id;

                // Check if there's a connection in either direction
                var hasConnection = false;

                // Check node1 -> node2
                if (connections[node1] && connections[node1][node2]) {{
                    hasConnection = true;
                }}

                // Check node2 -> node1
                if (connections[node2] && connections[node2][node1]) {{
                    hasConnection = true;
                }}

                // Only add line if there's a connection
                if (hasConnection) {{
                    greyLines.push({{
                        x: [nodes[i].x, nodes[j].x],
                        y: [nodes[i].y, nodes[j].y]
                    }});
                }}
            }}
        }}

        var greyLinesTrace = {{
            type: 'scatter',
            mode: 'lines',
            x: greyLines.flatMap(line => [line.x[0], line.x[1], null]),
            y: greyLines.flatMap(line => [line.y[0], line.y[1], null]),
            line: {{
                color: 'rgba(200, 200, 200, 0.3)',
                width: 3
            }},
            hoverinfo: 'skip',
            showlegend: false
        }};

        // Create location boxes at the bottom
        var locationBoxes = [];
        var boxWidth = 0.4;
        var boxHeight = 0.15;
        var boxY = -1.05;
        var spacing = 0.05;
        var totalWidth = locationNames.length * boxWidth + (locationNames.length - 1) * spacing;
        var startX = -totalWidth / 2 + boxWidth / 2;

        locationNames.forEach((loc, idx) => {{
            var boxX = startX + idx * (boxWidth + spacing);

            var boxShape = {{
                type: 'rect',
                x0: boxX - boxWidth / 2,
                x1: boxX + boxWidth / 2,
                y0: boxY - boxHeight / 2,
                y1: boxY + boxHeight / 2,
                fillcolor: locationColors[loc],
                opacity: 0.7,
                line: {{
                    color: 'white',
                    width: 2
                }},
                layer: 'below'
            }};

            locationBoxes.push(boxShape);
        }});

        // Create location box hover/click detection using invisible rectangles
        var locationHoverTraces = [];
        locationNames.forEach((loc, idx) => {{
            var boxX = startX + idx * (boxWidth + spacing);

            // Create grid of points covering the box for better hover detection
            var gridX = [];
            var gridY = [];
            for (var i = 0; i < 5; i++) {{
                for (var j = 0; j < 3; j++) {{
                    gridX.push(boxX - boxWidth/2 + (i * boxWidth/4));
                    gridY.push(boxY - boxHeight/2 + (j * boxHeight/2));
                }}
            }}

            locationHoverTraces.push({{
                type: 'scatter',
                x: gridX,
                y: gridY,
                mode: 'markers',
                marker: {{
                    size: 15,
                    color: 'rgba(0,0,0,0)',
                    opacity: 0
                }},
                hoverinfo: 'text',
                hovertext: loc,
                customdata: gridX.map(() => loc),
                showlegend: false
            }});
        }});

        // Create location text labels
        var locationTextTraces = [];
        locationNames.forEach((loc, idx) => {{
            var boxX = startX + idx * (boxWidth + spacing);

            locationTextTraces.push({{
                type: 'scatter',
                x: [boxX],
                y: [boxY],
                mode: 'text',
                text: [loc],
                textfont: {{
                    size: 12,
                    color: 'white',
                    family: 'Arial Black'
                }},
                hoverinfo: 'skip',
                showlegend: false
            }});
        }});

        // Create node traces by type - EXPLICIT ORDER with Z last
        var nodesByType = {{'A': [], 'B': [], 'Z': []}};
        nodes.forEach(node => {{
            nodesByType[node.type].push(node);
        }});

        var nodeTraces = [];
        var nodeTraceStartIndex = 1;
        var typeOrder = ['A', 'B', 'Z']; // Explicit order: Z last means Z on top
        typeOrder.forEach(type => {{
            var nodesOfType = nodesByType[type];
            if (nodesOfType.length > 0) {{
                nodeTraces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: nodesOfType.map(n => n.x),
                    y: nodesOfType.map(n => n.y),
                    text: nodesOfType.map(n => n.id),
                    textposition: 'middle center',
                    textfont: {{
                        size: 10,
                        color: 'white',
                        family: 'Arial Black'
                    }},
                    marker: {{
                        size: 35,
                        color: colors[type],
                        line: {{
                            color: 'white',
                            width: 2
                        }}
                    }},
                    hoverinfo: 'none',
                    customdata: nodesOfType.map(n => n.id),
                    showlegend: false
                }});
            }}
        }});

        var locationTraceStartIndex = 1 + nodeTraces.length;

        var layout = {{
            showlegend: false,
            xaxis: {{
                range: [-1.3, 1.3],
                showgrid: false,
                zeroline: false,
                showticklabels: false
            }},
            yaxis: {{
                range: [-1.3, 1.3],
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                scaleanchor: 'x',
                scaleratio: 1
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            hovermode: 'closest',
            margin: {{ l: 40, r: 40, t: 40, b: 40 }},
            shapes: locationBoxes
        }};

        // Initialize plot
        var allTraces = [greyLinesTrace, ...locationHoverTraces, ...locationTextTraces, ...nodeTraces];

        Plotly.newPlot('plotDiv', allTraces, layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false
        }});

        var plotDiv = document.getElementById('plotDiv');
        var currentHoveredMember = null;
        var currentHoveredLocation = null;
        var isHovering = false;
        var isPinned = false;
        var pinnedMember = null;
        var pinnedLocation = null;
        var lastHoverTime = 0;
        var hoverDebounceDelay = 200;

        // Table zoom functionality
        var currentZoom = 1.0;
        var tableContent = document.getElementById('tableContent');
        var zoomLevel = document.getElementById('zoomLevel');

        document.getElementById('zoomIn').addEventListener('click', function() {{
            currentZoom = Math.min(currentZoom + 0.1, 2.0);
            updateZoom();
        }});

        document.getElementById('zoomOut').addEventListener('click', function() {{
            currentZoom = Math.max(currentZoom - 0.1, 0.5);
            updateZoom();
        }});

        document.getElementById('resetTable').addEventListener('click', function() {{
            currentZoom = 1.0;
            updateZoom();
            document.getElementById('tableWrapper').scrollTop = 0;
            document.getElementById('tableWrapper').scrollLeft = 0;
        }});

        function updateZoom() {{
            tableContent.style.transform = 'scale(' + currentZoom + ')';
            zoomLevel.textContent = Math.round(currentZoom * 100) + '%';
        }}

        // Download CSV functionality
        document.getElementById('downloadCSV').addEventListener('click', function() {{
            var table = document.getElementById('dataTable');
            var csv = [];
            var rows = table.querySelectorAll('tr');

            for (var i = 0; i < rows.length; i++) {{
                var row = [], cols = rows[i].querySelectorAll('td, th');
                for (var j = 0; j < cols.length; j++) {{
                    row.push('"' + cols[j].textContent.replace(/"/g, '""') + '"');
                }}
                csv.push(row.join(','));
            }}

            var csvFile = new Blob([csv.join('\\n')], {{ type: 'text/csv' }});
            var downloadLink = document.createElement('a');
            downloadLink.download = 'collaboration_data.csv';
            downloadLink.href = window.URL.createObjectURL(csvFile);
            downloadLink.style.display = 'none';
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        }});

        // Function to highlight table cells
        function highlightTableMember(member) {{
            clearTableHighlights();

            if (!member) return;

            var rowHeaders = document.querySelectorAll('.row-header[data-member="' + member + '"]');
            rowHeaders.forEach(el => el.classList.add('highlighted'));

            var colHeaders = document.querySelectorAll('.col-header[data-member="' + member + '"]');
            colHeaders.forEach(el => el.classList.add('highlighted'));

            var rowCells = document.querySelectorAll('td[data-row="' + member + '"]');
            rowCells.forEach(el => el.classList.add('highlighted'));

            var colCells = document.querySelectorAll('td[data-col="' + member + '"]');
            colCells.forEach(el => el.classList.add('highlighted'));
        }}

        function highlightTableMembers(members) {{
            clearTableHighlights();

            members.forEach(member => {{
                var rowHeaders = document.querySelectorAll('.row-header[data-member="' + member + '"]');
                rowHeaders.forEach(el => el.classList.add('highlighted'));

                var colHeaders = document.querySelectorAll('.col-header[data-member="' + member + '"]');
                colHeaders.forEach(el => el.classList.add('highlighted'));

                var rowCells = document.querySelectorAll('td[data-row="' + member + '"]');
                rowCells.forEach(el => el.classList.add('highlighted'));

                var colCells = document.querySelectorAll('td[data-col="' + member + '"]');
                col
    
    
    
    
    
    function clearTableHighlights() {{
            var highlighted = document.querySelectorAll('.highlighted');
            highlighted.forEach(el => el.classList.remove('highlighted'));
        }}

        // Function to highlight location boxes
        function highlightLocationBoxes(locations) {{
            var newShapes = locationBoxes.map((box, idx) => {{
                var loc = locationNames[idx];
                var isHighlighted = locations.includes(loc);

                return {{
                    ...box,
                    opacity: isHighlighted ? 1.0 : 0.3,
                    line: {{
                        color: isHighlighted ? 'yellow' : 'white',
                        width: isHighlighted ? 4 : 2
                    }}
                }};
            }});

            return newShapes;
        }}

        function highlightNodesForLocation(location) {{
            var membersToHighlight = locationToMembers[location] || [];

            // Update node markers
            var updatedNodeTraces = nodeTraces.map(trace => {{
                var newMarkerSizes = [];
                var newLineWidths = [];

                for (var i = 0; i < trace.customdata.length; i++) {{
                    var memberId = trace.customdata[i];
                    var isHighlighted = membersToHighlight.includes(memberId);

                    if (isHighlighted) {{
                        newMarkerSizes.push(45);
                        newLineWidths.push(4);
                    }} else {{
                        newMarkerSizes.push(35);
                        newLineWidths.push(2);
                    }}
                }}

                return {{
                    ...trace,
                    marker: {{
                        ...trace.marker,
                        size: newMarkerSizes,
                        line: {{
                            color: 'white',
                            width: newLineWidths
                        }}
                    }}
                }};
            }});

            return updatedNodeTraces;
        }}

        // Helper function to create arrowhead path
        function createArrowhead(x, y, angle, size, color) {{
            var path = 'M ' + x + ' ' + y + 
                      ' L ' + (x - size * Math.cos(angle - Math.PI/6)) + ' ' + (y - size * Math.sin(angle - Math.PI/6)) +
                      ' L ' + (x - size * Math.cos(angle + Math.PI/6)) + ' ' + (y - size * Math.sin(angle + Math.PI/6)) +
                      ' Z';

            return {{
                type: 'path',
                path: path,
                fillcolor: color,
                line: {{color: color}},
                layer: 'below'
            }};
        }}

        // Function to redraw plot with connections using shapes for arrows
        function showConnections(hoveredMember) {{
            var memberConnections = connections[hoveredMember] || {{}};
            var hoveredNode = nodes.find(n => n.id === hoveredMember);

            var connectionTraces = [];
            var arrowShapes = [];

            var offsetDistance = 0.0075;
            var arrowSize = 0.025;

            Object.keys(memberConnections).forEach(targetMemberId => {{
                var conn = memberConnections[targetMemberId];
                var targetNode = nodes.find(n => n.id === targetMemberId);

                if (targetNode) {{
                    var hasOutgoing = conn.outgoing !== null;
                    var hasIncoming = conn.incoming !== null;
                    var isBidirectional = hasOutgoing && hasIncoming;

                    var dx = targetNode.x - hoveredNode.x;
                    var dy = targetNode.y - hoveredNode.y;
                    var dist = Math.sqrt(dx*dx + dy*dy);
                    var angle = Math.atan2(dy, dx);

                    var perpX = -dy / dist * offsetDistance;
                    var perpY = dx / dist * offsetDistance;

                    if (isBidirectional) {{
                        // Line 1: from hovered to target (offset)
                        connectionTraces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [hoveredNode.x + perpX, targetNode.x + perpX],
                            y: [hoveredNode.y + perpY, targetNode.y + perpY],
                            line: {{
                                color: colors[hoveredNode.type],
                                width: 2.5
                            }},
                            hoverinfo: 'skip',
                            showlegend: false
                        }});

                        // Arrow 1: pointing to target
                        var arrowX1 = hoveredNode.x + dx * 0.75 + perpX;
                        var arrowY1 = hoveredNode.y + dy * 0.75 + perpY;
                        arrowShapes.push(createArrowhead(arrowX1, arrowY1, angle, arrowSize, colors[hoveredNode.type]));

                        // Line 2: from target to hovered (opposite offset)
                        connectionTraces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [targetNode.x - perpX, hoveredNode.x - perpX],
                            y: [targetNode.y - perpY, hoveredNode.y - perpY],
                            line: {{
                                color: colors[targetNode.type],
                                width: 2.5
                            }},
                            hoverinfo: 'skip',
                            showlegend: false
                        }});

                        // Arrow 2: pointing to hovered
                        var arrowX2 = targetNode.x - dx * 0.75 - perpX;
                        var arrowY2 = targetNode.y - dy * 0.75 - perpY;
                        arrowShapes.push(createArrowhead(arrowX2, arrowY2, angle + Math.PI, arrowSize, colors[targetNode.type]));

                    }} else if (hasOutgoing) {{
                        // Single line and arrow (hovered -> target)
                        connectionTraces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [hoveredNode.x, targetNode.x],
                            y: [hoveredNode.y, targetNode.y],
                            line: {{
                                color: colors[hoveredNode.type],
                                width: 2.5
                            }},
                            hoverinfo: 'skip',
                            showlegend: false
                        }});

                        var arrowX = hoveredNode.x + dx * 0.8;
                        var arrowY = hoveredNode.y + dy * 0.8;
                        arrowShapes.push(createArrowhead(arrowX, arrowY, angle, arrowSize, colors[hoveredNode.type]));

                    }} else if (hasIncoming) {{
                        // Single line and arrow (target -> hovered)
                        connectionTraces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [targetNode.x, hoveredNode.x],
                            y: [targetNode.y, hoveredNode.y],
                            line: {{
                                color: colors[targetNode.type],
                                width: 2.5
                            }},
                            hoverinfo: 'skip',
                            showlegend: false
                        }});

                        var arrowX = targetNode.x - dx * 0.8;
                        var arrowY = targetNode.y - dy * 0.8;
                        arrowShapes.push(createArrowhead(arrowX, arrowY, angle + Math.PI, arrowSize, colors[targetNode.type]));
                    }}
                }}
            }});

            // Highlight location boxes for this member
            var memberLocs = memberLocations[hoveredMember] || [];
            var highlightedBoxes = highlightLocationBoxes(memberLocs);

            // Combine all shapes: location boxes + arrow shapes
            var allShapes = [...highlightedBoxes, ...arrowShapes];

            var newTraces = [...connectionTraces, ...locationHoverTraces, ...locationTextTraces, ...nodeTraces];
            var newLayout = {{
                ...layout,
                shapes: allShapes
            }};

            Plotly.react('plotDiv', newTraces, newLayout);
        }}

        function showLocationHighlight(location) {{
            var membersToHighlight = locationToMembers[location] || [];

            // Highlight the location box
            var highlightedBoxes = highlightLocationBoxes([location]);

            // Update node traces to highlight members
            var updatedNodeTraces = highlightNodesForLocation(location);

            var newTraces = [greyLinesTrace, ...locationHoverTraces, ...locationTextTraces, ...updatedNodeTraces];
            var newLayout = {{
                ...layout,
                shapes: highlightedBoxes
            }};

            Plotly.react('plotDiv', newTraces, newLayout);

            // Highlight table for these members
            highlightTableMembers(membersToHighlight);
        }}

        function showDefaultState() {{
            var defaultTraces = [greyLinesTrace, ...locationHoverTraces, ...locationTextTraces, ...nodeTraces];
            var defaultLayout = {{
                ...layout,
                shapes: locationBoxes
            }};
            Plotly.react('plotDiv', defaultTraces, defaultLayout);
        }}

        function updatePinNotice() {{
            var pinNotice = document.getElementById('pinNotice');
            if (isPinned) {{
                if (pinnedMember) {{
                    pinNotice.textContent = '📌 Pinned: ' + pinnedMember + ' (click to unpin)';
                }} else if (pinnedLocation) {{
                    pinNotice.textContent = '📌 Pinned: ' + pinnedLocation + ' (click to unpin)';
                }}
            }} else {{
                pinNotice.textContent = '';
            }}
        }}

        // Determine which trace type was clicked/hovered
        function getTraceType(curveNumber) {{
            if (curveNumber === 0) return 'greyLines';
            if (curveNumber >= 1 && curveNumber <= locationHoverTraces.length) return 'location';
            if (curveNumber > locationHoverTraces.length && curveNumber <= locationHoverTraces.length + locationTextTraces.length) return 'locationText';
            return 'node';
        }}

        // Click handler for pinning
        plotDiv.on('plotly_click', function(data) {{
            var point = data.points[0];
            var traceType = getTraceType(point.curveNumber);

            if (traceType === 'greyLines' || traceType === 'locationText') return;

            if (traceType === 'node') {{
                var clickedMember = point.customdata;

                if (isPinned && pinnedMember === clickedMember) {{
                    isPinned = false;
                    pinnedMember = null;
                    pinnedLocation = null;
                    showDefaultState();
                    clearTableHighlights();
                    updatePinNotice();
                }} else {{
                    isPinned = true;
                    pinnedMember = clickedMember;
                    pinnedLocation = null;
                    showConnections(clickedMember);
                    highlightTableMember(clickedMember);
                    updatePinNotice();
                }}
            }} else if (traceType === 'location') {{
                var clickedLocation = point.customdata;

                if (isPinned && pinnedLocation === clickedLocation) {{
                    isPinned = false;
                    pinnedMember = null;
                    pinnedLocation = null;
                    showDefaultState();
                    clearTableHighlights();
                    updatePinNotice();
                }} else {{
                    isPinned = true;
                    pinnedMember = null;
                    pinnedLocation = clickedLocation;
                    showLocationHighlight(clickedLocation);
                    updatePinNotice();
                }}
            }}
        }});

        // Hover handlers for plot
        plotDiv.on('plotly_hover', function(data) {{
            if (isPinned) return;

            lastHoverTime = Date.now();

            var point = data.points[0];
            var traceType = getTraceType(point.curveNumber);

            if (traceType === 'greyLines' || traceType === 'locationText') return;

            if (traceType === 'node') {{
                var hoveredMember = point.customdata;
                if (hoveredMember && hoveredMember !== currentHoveredMember) {{
                    currentHoveredMember = hoveredMember;
                    currentHoveredLocation = null;
                    isHovering = true;
                    showConnections(hoveredMember);
                    highlightTableMember(hoveredMember);
                }} else if (hoveredMember === currentHoveredMember) {{
                    isHovering = true;
                }}
            }} else if (traceType === 'location') {{
                var hoveredLocation = point.customdata;
                if (hoveredLocation && hoveredLocation !== currentHoveredLocation) {{
                    currentHoveredLocation = hoveredLocation;
                    currentHoveredMember = null;
                    isHovering = true;
                    showLocationHighlight(hoveredLocation);
                }} else if (hoveredLocation === currentHoveredLocation) {{
                    isHovering = true;
                }}
            }}
        }});

        plotDiv.on('plotly_unhover', function(data) {{
            if (isPinned) return;

            isHovering = false;

            setTimeout(function() {{
                var timeSinceLastHover = Date.now() - lastHoverTime;
                if (!isHovering && !isPinned && timeSinceLastHover > hoverDebounceDelay) {{
                    currentHoveredMember = null;
                    currentHoveredLocation = null;
                    showDefaultState();
                    clearTableHighlights();
                }}
            }}, hoverDebounceDelay);
        }});

        // Handle window resize
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('plotDiv');
        }});
    </script> </body> </html> '''

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Collaboration network with table created: {output_file}")
    print(f"📊 Loaded {len(members)} members")
    print(f"📍 Location data processed")

    # Print connection summary
    total_connections = 0
    bidirectional = 0
    for member in connections:
        for target in connections[member]:
            conn = connections[member][target]
            total_connections += 1
            if conn['outgoing'] and conn['incoming']:
                bidirectional += 1

    print(f"🔗 Total connections: {total_connections}")
    print(f"⟷ Bidirectional connections: {bidirectional}")

    # Print location summary
    print(f"\n📍 Members by location:")
    for loc in ['Berlin', 'Mainz', 'Munich', 'Würzburg']:
        count = sum(1 for member, locs in member_locations.items() if loc in locs)
        print(f"   {loc}: {count} members")