from .utils import *
from .forecaster import *
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import webbrowser

@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )

def run_server(branch):
    properties = branch.properties.propnum.unique()
    if branch.forecaster is None:
        print('true')
        branch.forecast()

    i = 0
    prod_type = 'gas'

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    idp = properties[i]
    well = branch.model[idp]
    production = load_production(branch, well.idp)
    forecast = load_forecast(branch.forecaster, idp,
                                well.forecasts.prod_forecast_scenario, t_start=0, t_end=18250)

    prod_start_date = production.prod_date.min()
    last_prod_date = production.prod_date.max()

    fig = go.Figure()
    fig.update_yaxes(type='log')
    fig.update_layout(xaxis_title='date',
                      yaxis_title=prod_type + ' production',
                      height=1000,
                      font={'size': 18})

    if forecast.prod_date[0]:
        idf = forecast[forecast.prod_date == last_prod_date].index[0]
        fig.add_trace(go.Scatter(x=forecast.prod_date[idf:idf+500],
                                 y=forecast[prod_type][idf:idf+500],
                                 line={'color': 'red'}, name='forecast'))
    else:
        if p is not None:
            x_range = pd.date_range(start=prod_start_date, periods=1000, freq='d')
            fig.add_trace(go.Scatter(x=x_range,
                                     y=forecast[prod_type][:1000],
                                     line={'color': 'red'}, name='forecast'))
        else:
            fig.add_trace(go.Scatter(x=forecast['time_on'][:1000],
                                     y=forecast[prod_type][:1000],
                                     line={'color': 'red'}, name='forecast'))

    if production.shape[0] <= 30:
        window = 3
    elif production.shape[0] > 30 and production.shape[0] <= 240:
        window = 7
    elif production.shape[0] > 240 and production.shape[0] <= 540:
        window = 14
    else:
        window = 30
    prod_smooth = production[prod_type].rolling(window, center=True).mean()

    fig.add_trace(go.Scatter(x=production.prod_date,
                             y=prod_smooth, name='smoothed production',
                             line={'color': 'black'}, opacity=0.75))

    fig.add_trace(go.Scatter(x=production.prod_date,
                             y=production[prod_type],
                             line={'color': 'slategray'}, name='production'))

    app.layout = html.Div(children=[
        html.H1(children='Dash'),
        html.Div(dcc.Input(id='input-on-submit', type='text')),
        html.Button('Submit', id='submit-val', n_clicks=0),
        html.Div(id='container-button-basic', children='Enter a value and press submit'),
        dcc.Graph(id='example-graph', figure=fig)
        ])

    webbrowser.open_new('http://localhost:{}'.format(8050))

    app.run_server()

