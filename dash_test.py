import olive
from olive.utils import *
from olive.forecaster import *
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import webbrowser
import sys
from datetime import date

class plot():
    def __init__(self, branch, properties, idx=None, idp=None):
        self.branch = branch
        self.tree = branch.tree
        self.properties = properties
        self.well_dropdown = []
        self.num_prop = len(self.properties)
        self.idx = idx
        self.idp = idp
        self.well = None
        self.prod_type = 'gas'
        self.tmp_prod_info = None
        self.tmp_yields_dict = None
        self.tmp_prod_fcst = None
        self.figure = None
        self.new_fit = None
        self.new_hindcast = None
        self.initialized = False
        self.config = {'toImageButtonOptions': {'width': 900, 'height': 600}}
        if not self.initialized:
            self.build_well_dropdown()
            self.get_idp()
            self.get_well()
            self.build_figure()
            self.initialized = True

    def get_idp(self):
        if self.idx is not None:
            try: 
                self.idx = int(self.idx)
                self.idp = properties[self.idx]
            except:
                print('bad idx')
                sys.stdout.flush()

    def get_idx(self):
        if self.idp is not None:
            if type(self.idp) == str:
                self.idx = np.where(properties == self.idp)[0][0]
            else:
                print('bad idp')
                self.idp = None

    def get_well(self):
        if self.idp is not None:
            self.production = None
            self.prod_start_date = None
            self.last_prod_date = None
            self.new_fit = None
            self.new_hindcast = None
            self.tmp_prod_info = None
            self.tmp_yields_dict = None
            self.tmp_prod_fcst = None
            self.forecast_type = None
            self.type_curve = None
            self.well = self.branch.model[self.idp]
            self.prod_info = self.well.forecasts.prod_info[self.prod_type]
            if self.prod_info:
                self.forecast_type = self.prod_info['forecast_type'][0]
                self.type_curve = self.prod_info['type_curve'][0]
                self.start_date = pd.Timestamp(self.prod_info['fcst_start_date'][0]).date()
                self.b = self.prod_info['b_factor'][0]
                self.di = self.prod_info['initial_decline'][0]
                self.ip = self.prod_info['initial_rate'][0]
                if self.prod_info['prod_shift'][0] > 0:
                    self.ip = self.prod_info['initial_rate'][0] * self.prod_info['prod_shift'][0]
                self.dmin = self.prod_info['terminal_decline'][0]
                self.min_rate = self.prod_info['min_rate'][0]
                self.ratio = self.prod_info['ratio'][0]
                self.tmp_prod_info = self.prod_info.copy()
                self.tmp_yields_dict = self.well.forecasts.yields_dict.copy()
            else:
                self.start_date = date.today()
                self.b = 1.05
                self.di = 0.80
                self.ip = 5000
                self.dmin = 0.06
                self.min_rate = 10
                self.forecast_type = self.well.forecasts.forecast_type
                self.type_curve = self.well.forecasts.forecast
            self.yields_placeholders(use_tmp=True)

    def build_well_dropdown(self):
        for _, row in self.branch.properties.iterrows():
            self.well_dropdown.append({'label': row['bolo_well_name'],
                                       'value': row['propnum']})

    def yields_placeholders(self, use_tmp=False):
        if use_tmp:
            if self.tmp_yields_dict is not None:
                for k, v in self.tmp_yields_dict.items():
                    if k == 'gas':
                        if v is None:
                            self.gas_yield_placeholder = 0.0
                        else:
                            self.gas_yield_placeholder = v
                    if k == 'oil':
                        if v is None:
                            self.oil_yield_placeholder = 0.0
                        else:
                            self.oil_yield_placeholder = v
                    if k == 'water':
                        if v is None:
                            self.water_yield_placeholder = 0.0
                        else:
                            self.water_yield_placeholder = v
            else:
                self.gas_yield_placeholder = 0.0
                self.oil_yield_placeholder = 0.0
                self.water_yield_placeholder = 0.0
        else:
            if self.well.forecasts.yields_dict is not None:
                for k, v in self.well.forecasts.yields_dict.items():
                    if k == 'gas':
                        if v is None:
                            self.gas_yield_placeholder = 0.0
                        else:
                            self.gas_yield_placeholder = v
                    if k == 'oil':
                        if v is None:
                            self.oil_yield_placeholder = 0.0
                        else:
                            self.oil_yield_placeholder = v
                    if k == 'water':
                        if v is None:
                            self.water_yield_placeholder = 0.0
                        else:
                            self.water_yield_placeholder = v
            else:
                self.gas_yield_placeholder = 0.0
                self.oil_yield_placeholder = 0.0
                self.water_yield_placeholder = 0.0

    def get_hindcasts(self, use_tmp=False):
        if use_tmp:
            start_date = pd.Timestamp(p.tmp_prod_info['fcst_start_date'][0])
            b = p.tmp_prod_info['b_factor'][0]
            di = p.tmp_prod_info['initial_decline'][0]
            ip = p.tmp_prod_info['initial_rate'][0]
            dmin = p.tmp_prod_info['terminal_decline'][0]
            min_rate = p.tmp_prod_info['min_rate'][0]
            forecast = arps_fit([b, di, ip], dmin, min_rate)
        else:
            start_date = pd.Timestamp(self.prod_info['fcst_start_date'][0])
            forecast = arps_fit([self.b, self.di, self.ip], self.dmin, self.min_rate)
        prod_date = pd.date_range(start_date,
                                  periods=self.idf, freq='D')
        time_on = time_on = np.arange(1, self.idf)
        return forecast, prod_date, time_on

    def build_figure(self):
        if self.idp is not None:
            self.production = load_production(self.branch, self.idp)
            forecast = load_forecast(self.branch.forecaster, self.idp,
                                     self.well.forecasts.prod_forecast_scenario,
                                     t_start=0, t_end=18250)

            self.prod_start_date = self.production.prod_date.min()
            self.last_prod_date = self.production.prod_date.max()
            try:
                d = (self.last_prod_date - self.prod_start_date).days
            except:
                d = None
            
            fig = go.Figure()
            
            if len(forecast.prod_date) > 0:
                self.idf = forecast[forecast.prod_date == self.last_prod_date].index[0]
                sys.stdout.flush()
                fig.add_trace(go.Scatter(x=forecast.prod_date[self.idf:],
                                        y=forecast[self.prod_type][self.idf:],
                                        line={'color': 'red'}, name='forecast'))
                if self.forecast_type in ('auto', 'manual'):
                    print(self.forecast_type, self.type_curve, 'auto/manual forecast')
                    sys.stdout.flush()
                    hindcast, back_dates, time_on = self.get_hindcasts(use_tmp=False)
                    fig.add_trace(go.Scatter(x=back_dates,
                                             y=hindcast,
                                             line={'color': 'red',
                                                   'dash': 'dash'}, opacity=0.5, name='hindcast'))   
                else:
                    print(self.forecast_type, self.type_curve, 'autotype forecast')
                    sys.stdout.flush()
                    hindcast = load_type_curve(self.well, self.type_curve)
                    hindcast = hindcast[:self.idf]
                    hindcast.loc[:, self.prod_type] = hindcast.loc[:, self.prod_type] * self.ratio
                    fig.add_trace(go.Scatter(x=forecast.prod_date[:self.idf],
                                             y=hindcast[self.prod_type],
                                             line={'color': 'red',
                                                   'dash': 'dash'}, opacity=0.5, name='hindcast'))
                fig.update_xaxes(range=[forecast.prod_date[0], forecast.prod_date[self.idf+500]])

            else:
                print(self.forecast_type, self.type_curve, 'type forecast')
                if d is not None:
                    x_range = pd.date_range(start=self.prod_start_date, periods=18250, freq='d')
                    fig.add_trace(go.Scatter(x=x_range,
                                            y=forecast[self.prod_type],
                                            line={'color': 'red'}, name='type curve forecast', showlegend=True))
                    fig.update_xaxes(range=[x_range[0], x_range[1000]])
                else:
                    fig.add_trace(go.Scatter(x=forecast['time_on'],
                                            y=forecast[self.prod_type],
                                            line={'color': 'red'}, name='type curve forecast', showlegend=True))
                    fig.update_xaxes(range=[forecast['time_on'][0], forecast['time_on'][1000]])
            if d is not None:
                if self.production.shape[0] <= 30:
                    window = 3
                elif self.production.shape[0] > 30 and self.production.shape[0] <= 240:
                    window = 7
                elif self.production.shape[0] > 240 and self.production.shape[0] <= 540:
                    window = 14
                else:
                    window = 30
                prod_smooth = self.production[self.prod_type].rolling(window, center=True).mean()
                sys.stdout.flush()
                fig.add_trace(go.Scatter(x=self.production.prod_date,
                                        y=prod_smooth, name='smoothed production',
                                        line={'color': 'black'}, opacity=0.7))

                fig.add_trace(go.Scatter(x=self.production.prod_date,
                                        y=self.production[self.prod_type],
                                        line={'color': 'slategray'}, name='production'))
            
            if self.new_fit is not None:
                fig.add_trace(self.new_fit)
            if self.new_hindcast is not None:
                fig.add_trace(self.new_hindcast)

            if len(forecast.prod_date) > 0:
                eur = self.tmp_prod_info['eur'][0]
                textstr = '<br>'.join(('idp: ' + self.idp,
                                    'name: ' + self.well.well_name,
                                    'pad: ' + self.well.short_pad,
                                    'pod: ' + self.well.pad,
                                    'prod type: ' + self.prod_type,
                                    'date of first production: '+ self.prod_start_date.strftime('%m/%d/%Y'),
                                    'date of last production: ' + self.last_prod_date.strftime('%m/%d/%Y'),
                                    'production length: ' + str(d) + ' days',
                                    'eur: ' + '{:,.0f}'.format(eur)))
            else:
                eur = forecast[self.prod_type].sum()
                textstr = '<br>'.join(('idp: ' + self.idp,
                                    'name: ' + self.well.well_name,
                                    'pad: ' + self.well.short_pad,
                                    'pod: ' + self.well.pad,
                                    'prod type: ' + self.prod_type,
                                    'type curve: ' + self.type_curve,
                                    'eur: ' + '{:,.0f}'.format(eur)))


            sys.stdout.flush()
            fig.update_yaxes(type='log')
            fig.update_layout(xaxis_title='date',
                              yaxis_title=self.prod_type + ' production',
                              height=600,
                              font={'family': 'Arial Rounded MT',
                                    'size': 16},
                              dragmode='pan',
                              annotations=[
                                  dict(
                                      xref='paper',
                                      yref='paper',
                                      x=1.0,
                                      y=0,
                                      xshift=200,
                                      text=textstr,
                                      showarrow=False,
                                      align='left',
                                      yanchor='bottom',
                                      bgcolor='rgba(255, 2525, 255, .8)'
                                  )

                              ]
                                    
                                    
                                    )
            self.figure = fig


t = olive.Tree('output', create_folders=False, verbose=False)
print('loading temp load file')
sys.stdout.flush()
t.load_branch('temp\\load')
b = t.branches[list(t.branches.keys())[0]]
properties = b.properties.propnum.unique()
if b.forecaster is None:
    b.forecast()

p = plot(b, properties, idx=0)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)
app.title = 'olive beta'

app.layout = html.Div([
    html.Div(id='plot_container', children=dcc.Graph(id='plot', figure=p.figure, config=p.config)),
    html.P(id='idx', children=str(str(p.idx + 1) + ' of ' + str(p.num_prop)), style={'font-weight': 'bold'}),
    html.Button('previous', id='prev_click', n_clicks=0),
    html.Button('next', id='next_click', n_clicks=0),
    html.Div([dcc.Input(id='input_idp', type='text', placeholder='enter idp'),
              html.Button('plot', id='submit_idp', n_clicks=0)]),
    html.Div([
        html.Div([html.P('forecast start date:'),
                dcc.DatePickerSingle(id='input_start_date',
                                    date=p.start_date)]),
        html.Div([html.P('initial rate:'),
                dcc.Input(id='ip', type='number', min=1.0, placeholder=round(p.ip, 0))]),
        html.Div([html.P('b factor:'),
                dcc.Input(id='b', type='number', min=0.0, max=2.0, placeholder=round(p.b, 2))]),
        html.Div([html.P('initial decline:'),
                dcc.Input(id='di', type='number', min=0.1, max=0.99, placeholder=round(p.di, 2))]),
        html.Div([html.P('terminal decline:'),
                dcc.Input(id='dmin', type='number', min=0.05, max=0.25, placeholder=round(p.dmin, 2))]),
        html.Div([html.P('minimum rate:'),
                dcc.Input(id='min_rate', type='number', min=1.0, placeholder=round(p.min_rate, 0))]),
        html.Div([html.P('gas yield:'),
                dcc.Input(id='gas_yield', type='number', min=1.0, placeholder=round(p.gas_yield_placeholder, 1))]),
        html.Div([html.P('oil yield:'),
                dcc.Input(id='oil_yield', type='number', min=1.0, placeholder=round(p.oil_yield_placeholder, 1))]),
        html.Div([html.P('water yield:'),
                dcc.Input(id='water_yield', type='number', min=1.0, placeholder=round(p.water_yield_placeholder, 1))]),
        html.Button('update', id='submit_update', n_clicks=0),
        html.Button('clear', id='submit_clear', n_clicks=0),
        html.Button('save', id='submit_save', n_clicks=0),
        html.Label(['well name', dcc.Dropdown(id='well_dropdown',
            options=p.well_dropdown)], style={'width': '400px'})
    ])

    ])

@app.callback([Output('plot_container', 'children'),
               Output('idx', 'children'),
               Output('input_start_date', 'date'),
               Output('ip', 'placeholder'),
               Output('b', 'placeholder'),
               Output('di', 'placeholder'),
               Output('dmin', 'placeholder'),
               Output('min_rate', 'placeholder'),
               Output('gas_yield', 'placeholder'),
               Output('oil_yield', 'placeholder'),
               Output('water_yield', 'placeholder')
               ],
              [Input('prev_click', 'n_clicks'),
               Input('next_click', 'n_clicks'),
               Input('submit_idp', 'n_clicks'),
               Input('submit_update', 'n_clicks'),
               Input('submit_clear', 'n_clicks'),
               Input('submit_save', 'n_clicks'),
               Input('well_dropdown', 'value')],
              [State('input_idp', 'value'),
               State('input_start_date', 'date'),
               State('ip', 'value'),
               State('b', 'value'),
               State('di', 'value'),
               State('dmin', 'value'),
               State('min_rate', 'value'),
               State('gas_yield', 'value'),
               State('oil_yield', 'value'),
               State('water_yield', 'value')
               ])
def change_figure(prev_btn, next_btn, plot_btn, update_btn, clear_btn,  save_btn,
                  dropdown_idp, idp, start_date, ip, b, di, dmin, min_rate,
                  gas_yield, oil_yield, water_yield):
    print(prev_btn, next_btn, plot_btn, update_btn, clear_btn, save_btn,
          dropdown_idp, idp, start_date, ip, b, di, dmin, min_rate,
          gas_yield, oil_yield, water_yield)
    sys.stdout.flush()
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'next_click' in changed_id:
        print('next button clicked', p.idx, p.num_prop)
        sys.stdout.flush()
        if p.idx < p.num_prop:
            p.idx = p.idx + 1
            print(p.idx)
            sys.stdout.flush()
            p.get_idp()
            p.get_well()
            p.build_figure()
            start_date = p.start_date
    if 'prev_click' in changed_id:
        print('previous button clicked', p.idx, p.num_prop)
        sys.stdout.flush()
        if p.idx > 0:
            p.idx = p.idx - 1
            print(p.idx)
            sys.stdout.flush()
            p.get_idp()
            p.get_well()
            p.build_figure()
            start_date = p.start_date
    if 'submit_idp' in changed_id:
        print('submit button clicked', p.idx, p.num_prop, p.idp)
        sys.stdout.flush()
        if idp in p.properties:
            p.idp = idp
            p.get_idx()
            p.get_well()
            p.build_figure()
            start_date = p.start_date
    if 'submit_update' in changed_id:
        print('update button clicked')
        sys.stdout.flush()
        if start_date is None:
            start_date = p.start_date
        if b is None:
            b = p.b
        if di is None:
            di = p.di
        if ip is None:
            ip = p.ip
        if dmin is None:
            dmin = p.dmin
        if min_rate is None:
            min_rate = p.min_rate
        if p.tmp_prod_info is None:
            p.tmp_prod_info = p.prod_info.copy()
        if gas_yield is None:
            gas_yield = p.well.forecasts.yields.gas
            if gas_yield != gas_yield:
                gas_yield = 0.0
        if oil_yield is None:
            oil_yield = p.well.forecasts.yields.oil
            if oil_yield != oil_yield:
                oil_yield = 0.0
        if water_yield is None:
            water_yield = p.well.forecasts.yields.water
            if water_yield != water_yield:
                water_yield = 0.0
        p.tmp_prod_info['fcst_start_date'] = [start_date]
        p.tmp_prod_info['b_factor'] = [b]
        p.tmp_prod_info['initial_decline'] = [di]
        p.tmp_prod_info['initial_rate'] = [ip]
        p.tmp_prod_info['terminal_decline'] = [dmin]
        p.tmp_prod_info['min_rate'] = [min_rate]
        p.tmp_prod_info['forecast_type'] = ['manual']
        p.tmp_prod_info['type_curve'] = [p.idp]
        p.tmp_prod_info['prod_shift'] = [0.0]
        p.tmp_prod_info['rmse'] = [0.0]
        p.tmp_yields_dict = {'gas': gas_yield, 'oil': oil_yield, 'water': water_yield}
        p.yields_placeholders(use_tmp=False)
        if update_btn > 0:
            if (b > 0.0 and b < 2.0 and di >= 0.1 and di <= 0.99 and ip >= 1.0
                and dmin >= 0.05 and dmin <= 0.25 and min_rate >= 1.0):
                p.tmp_fcst_dict = manual_fit(p.well, start_date, p.production, p.tmp_yields_dict,
                                            p.prod_type, [b, di, ip], dmin, min_rate)
                p.tmp_prod_info['eur'] = [p.tmp_fcst_dict[p.prod_type].sum()]
                p.new_fit = go.Scatter(x=p.tmp_fcst_dict['prod_date'][p.idf:p.idf+500],
                                    y=p.tmp_fcst_dict[p.prod_type][p.idf:p.idf+500],
                                    line={'dash': 'dash',
                                        'color': 'purple'}, name='new forecast')
                if pd.Timestamp(start_date) < p.last_prod_date:
                    delta = (p.last_prod_date - pd.Timestamp(start_date)).days
                    hindcast, prod_date, time_on = p.get_hindcasts(use_tmp=True)
                    p.new_hindcast = go.Scatter(x=prod_date[:delta],
                                                y=hindcast[:delta],
                                                mode='lines',
                                                line={'dash': 'dash',
                                                    'color': 'purple'}, opacity=0.5, name='new hindcast')
                p.build_figure()
    if 'submit_clear' in changed_id:
        print('clear button clicked')
        sys.stdout.flush()
        p.new_fit = None
        p.get_well()
        p.build_figure()
    if 'submit_save' in changed_id:
        print('save button clicked')
        sys.stdout.flush()
        save_manual_prod_info(p.branch, p.tmp_prod_info, p.prod_type, p.well.idp)
        p.well.forecasts.yields_dict = p.tmp_yields_dict
        p.tmp_fcst_dict = manual_fit(p.well, start_date, p.production, p.tmp_yields_dict,
                                     p.prod_type, [p.tmp_prod_info['b_factor'][0],
                                                   p.tmp_prod_info['initial_decline'][0],
                                                   p.tmp_prod_info['initial_rate'][0]],
                                     p.tmp_prod_info['terminal_decline'][0], p.tmp_prod_info['min_rate'][0])
        save_manual_prod_forecast(p.branch, p.tmp_fcst_dict, p.well.idp)
        p.well.forecasts.prod_info[p.prod_type] = p.tmp_prod_info
        p.well.forecasts.yields = Well_Yields(p.well.forecasts, p.tmp_yields_dict)
        p.new_fit = None
        p.new_hindcast = None
        p.tmp_prod_fcst = None
        p.tmp_prod_info = None
        p.tmp_yields_dict = None
        p.get_well()
        p.build_figure()
    if 'well_dropdown' in changed_id:
        print('well picked', dropdown_idp)
        sys.stdout.flush()
        if dropdown_idp in p.properties:
            p.idp = dropdown_idp
            p.get_idx()
            p.get_well()
            p.build_figure()
            start_date = p.start_date

    return (dcc.Graph(id='plot', figure=p.figure, config=p.config),
            str(str(p.idx + 1) + ' of ' + str(p.num_prop)),
            start_date,
            round(p.ip, 0),
            round(p.b, 2),
            round(p.di, 2),
            round(p.dmin, 2),
            round(p.min_rate, 0),
            round(p.gas_yield_placeholder, 1),
            round(p.oil_yield_placeholder, 1),
            round(p.water_yield_placeholder, 1),
            )


def manual_fit(well, fcst_start_date, production, yields_dict,
               prod_type, params, dmin, min_rate):
    fcst_dict = {'scenario': [well.forecasts.prod_forecast_scenario] * 18250,
                 'idp': [well.idp] * 18250,
                 'forecast': [well.forecasts.forecast] * 18250,
                 'time_on': np.arange(1, 18251),
                 'prod_date': [None] * 18250,
                 'prod_cat': ['forecast'] * 18250,
                 'gas': np.zeros(18250),
                 'oil': np.zeros(18250),
                 'water': np.zeros(18250),
                 'cum_gas': np.zeros(18250),
                 'cum_oil': np.zeros(18250),
                 'cum_water': np.zeros(18250)}

    forecast = arps_fit(params, dmin, min_rate)

    max_idx = production[:120][prod_type].idxmax()
    max_rate = production[prod_type].max()
    try:
        max_date = production.loc[max_idx, 'prod_date']
    except:
        max_date = np.nan

    prod_time = production.shape[0]
    fcst_start_date = pd.Timestamp(fcst_start_date)
    start_idx = prod_time - (fcst_start_date - production.prod_date.min()).days - 2
    forecast = np.concatenate([production[prod_type].values[:-2], forecast[start_idx:]])
    forecast = forecast[:18250]
    fcst_dict[prod_type] = forecast
    fcst_dict[str('cum_' + prod_type)] = forecast.cumsum()

    for t, y in yields_dict.items():
        if y is not None and y == y:
            if t == prod_type:
                yields_dict[t] = None
            if t == 'gas':
                fcst_dict['gas'] = fcst_dict['oil'] * y
                fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()
            if t == 'oil':
                fcst_dict['oil'] = fcst_dict['gas'] * y / 1000
                fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()
            if t == 'water':
                fcst_dict['water'] = fcst_dict['gas'] * y / 1000
                fcst_dict['cum_water'] = fcst_dict['water'].cumsum()

    prod_date = pd.date_range(production.prod_date.min(), periods=18250, freq='D')
    fcst_dict['prod_date'] = prod_date

    prod_cat = ['actual'] * len(production[:start_idx])
    forecast_cat = ['forecast'] * (len(forecast) - len(production[:start_idx]))
    prod_cat.extend(forecast_cat)
    fcst_dict['prod_cat'] = prod_cat

    time_on = time_on = np.arange(1, 18251)
    fcst_dict['time_on'] = time_on

    return fcst_dict

if __name__ == '__main__':
    webbrowser.open_new('http://localhost:{}'.format(8050))
    app.run_server()