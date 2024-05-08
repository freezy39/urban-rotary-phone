# Import required libraries
import pandas as pd
import dash
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()


app = dash.Dash(__name__)

unique_sites = spacex_df['Launch Site'].unique()

site_dropdown = dcc.Dropdown(
    id='site-dropdown',
    options=[
        {'label': 'All Sites', 'value': 'ALL'}
    ] + [{'label': site, 'value': site} for site in unique_sites],
    value='ALL',
    placeholder="Select a Launch Site here",
    searchable=True
)

# app layout
app.layout = html.Div(children=[html.H2('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 50}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                site_dropdown,
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(
                                    id='payload-slider',
                                    min=0,  
                                    max=10000, 
                                    step=1000,
                                    marks={i: f'{i}' for i in range(0, 10001, 1000)},  # Marks on the slider
                                    value=[0, 10000]  # Initial value (range)
                                ),
                                
                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        
        fig = px.pie(
            spacex_df, 
            values='class', 
            names='Launch Site', 
            title='Total Success Launches by Site'
        )
    else:
        
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
        fig = px.pie(
            filtered_df, 
            names='class', 
            title=f'Total Success and Fail Launches for {selected_site}'
        )
    return fig
    
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter_chart(selected_site, payload_range):
    filtered_df = spacex_df[
        (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
        (spacex_df['Payload Mass (kg)'] <= payload_range[1])
    ]
    
    if selected_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == selected_site]
    
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title='Correlation between Payload and Success for selected site(s)'
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8054)
