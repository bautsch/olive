from .utils import *
from .forecaster import *
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

def plot(branch, properties=None):
    run = True
    if properties is None:
        properties = branch.properties.propnum.unique()
    if branch.forecaster is None:
        branch.forecast()
    print('\r')
    print('forecast mode')

    i = 0
    plot = True
    plot_cmd = None
    plot_lock = False
    tmp_prod_info = None
    prod_info = None
    tmp_fcst_dict = None
    prod = True
    fcst = True
    smooth = True
    prod_type = 'gas'

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    while plot == True:

        if plot_cmd == 'q':
            # plt.close()
            print('closing plot\n')
            plot = False

        elif plot_cmd == 'n':
            if not plot_lock:
                # plt.clf()
                prod = True
                fcst = True
                tmp_prod_info = None
            if i + 1 >= len(properties):
                print('end of property list')
            else:
                i += 1
                prod = True
                fcst = True
                tmp_prod_info = None

        elif plot_cmd == 'p':
            if not plot_lock:
                plt.clf()
                prod = True
                fcst = True
                tmp_prod_info = None
            if i - 1 < 0:
                print('beginning of list')
            else:
                i -= 1
                prod = True
                fcst = True
                tmp_prod_info = None

        elif plot_cmd in properties:
            if not plot_lock:
                plt.clf()
            i = np.where(properties == plot_cmd)[0][0]
            prod = True
            fcst = True
            tmp_prod_info = None

        elif plot_cmd == 'l':
            print('locking plot')
            plot_lock = True
            plt.clf()
            prod = True
            fcst = True
            tmp_prod_info = None

        elif plot_cmd == 'u':
            print('unlocking plot')
            plot_lock = False
            plt.clf()
            prod = True
            fcst = True
            tmp_prod_info = None

        elif plot_cmd == 'm':
            vals = input('parameter: ').split(' ')
            m_cmd = True
            input_params = {'prod_type': None,
                            'b_factor': None,
                            'initial_decline': None,
                            'initial_rate': None,
                            'fcst_start_date': None,
                            'terminal_decline': None,
                            'min_rate': None}

            for idv, v in enumerate(vals):
                if v in ('b', 'di', 'ip', 'dmin', 'min_rate'):
                    try:
                        value = vals[idv+1]
                    except:
                        print('missing value after parameter', v)
                        m_cmd = False
                        break

                    try:
                        value = float(value)
                        if v == 'b':
                            v = 'b_factor'
                        elif v == 'di':
                            v = 'initial_decline'
                        elif v == 'ip':
                            v = 'initial_rate'
                        elif v == 'dmin':
                            v = 'terminal_decline'
                        input_params[v] = value
                    except:
                        print(v, 'must be float')
                        m_cmd = False
                        break

                if v == 'start':
                    try:
                        value = vals[idv+1]
                    except:
                        print('missing value after parameter', v)
                        m_cmd = False
                        break
                    try:
                        value = pd.Timestamp(value)
                        v = 'fcst_start_date'
                        input_params[v] = value
                    except:
                        print(v, 'must be date, m/d/y')
                        m_cmd = False
                        break

                if v == 'prod':
                    try:
                        value = vals[idv+1]
                    except:
                        print('missing value after parameter', v)
                        m_cmd = False
                        break
                    if value in ('gas', 'oil', 'water'):
                        v = 'prod_type'
                        input_params[v] = value
                    else:
                        print(v, 'must be gas, oil, or water')
                        m_cmd = False
                        break

            if m_cmd:
                if all(v == None for v in input_params.values()):
                    fcst = False
                    continue

                for k, v in input_params.items():
                    if v is not None:
                        if tmp_prod_info is None:
                            tmp_prod_info = well.forecasts.prod_info[prod_type].copy()
                            tmp_prod_info[k] = [v]
                        else:
                            tmp_prod_info[k] = [v]

                tmp_prod_info['forecast_type'] = ['manual']
                tmp_prod_info['prod_shift'] = [0.0]
                tmp_prod_info['rmse'] = [0.0]
                fcst = True
                params = [tmp_prod_info['b_factor'][0],
                          tmp_prod_info['initial_decline'][0],
                          tmp_prod_info['initial_rate'][0]]
                dmin = tmp_prod_info['terminal_decline'][0]
                min_rate = tmp_prod_info['min_rate'][0]
                fcst_start_date = tmp_prod_info['fcst_start_date'][0]
                tmp_fcst_dict = manual_fit(well, fcst_start_date, production, prod_type,
                                            params, dmin, min_rate)
                tmp_prod_info['eur'] = [tmp_fcst_dict[prod_type].sum()]

        elif plot_cmd == 'r':
            print('refreshing plot\n')
            tmp_prod_info = None
            tmp_fcst_dict = None
            prod = True
            fcst = True
            # plt.clf()

        elif plot_cmd == 's':
            if tmp_prod_info is None:
                print('nothing to save')
            else:
                print('prod type', prod_type)
                for k, v in tmp_prod_info.items():
                    print(k, v)
                check = input('save last manual forecast (y/n)? ')
                if check == 'y':
                    save_manual_prod_info(branch, tmp_prod_info, prod_type, well.idp)
                    save_manual_prod_forecast(branch, tmp_fcst_dict, well.idp)
                    well.forecasts.prod_info[prod_type] = tmp_prod_info
                    tmp_prod_info = None
                    tmp_fcst_dict = None
                    prod = True
                    fcst = True
                    # plt.clf()
                else:
                    print('\n')
                    pass

        elif plot_cmd == 'gas':
            prod_type = 'gas'
            tmp_prod_info = None
            tmp_fcst_dict = None
            prod = True
            fcst = True
            plt.clf()

        elif plot_cmd == 'oil':
            prod_type = 'oil'
            tmp_prod_info = None
            tmp_fcst_dict = None
            prod = True
            fcst = True     
            plt.clf()
    
        elif plot_cmd is None:
            pass

        else:
            print('plot command not recognized\n')

        if plot:
            idp = properties[i]
            well = branch.model[idp]
            print(branch.tree)
            production = load_production(branch, well.idp)
            forecast = load_forecast(branch.forecaster, idp,
                                        well.forecasts.prod_forecast_scenario, t_start=0, t_end=18250)

            print(i+1, 'of', len(properties))
            print('idp:', idp)
            print('name:', well.well_name)
            print('pad:', well.pad)
            print('short pad:', well.short_pad)
            print('prod type:', prod_type)

            if tmp_prod_info is None:
                prod_info = well.forecasts.prod_info[prod_type]
            else:
                prod_info = tmp_prod_info

            prod_start_date = production.prod_date.min()
            last_prod_date = production.prod_date.max()
            try:
                p = (last_prod_date - prod_start_date).days
                print('\r')
                print('date of first production:',
                        prod_start_date.strftime('%m/%d/%Y'))
                print('date of last production:',
                        last_prod_date.strftime('%m/%d/%Y'))
                print('production length:', p, 'days')
            except:
                p = None
                print('\r')
                print('no production history')

            if prod_info:
                fcst_start_date = pd.Timestamp(prod_info['fcst_start_date'][0])
                print('\r')
                print('forecast start date',
                        fcst_start_date.strftime('%m/%d/%Y'))
                print('b:', round(prod_info['b_factor'][0], 2))
                print('di:', round(prod_info['initial_decline'][0], 2))
                print('ip:', round(prod_info['initial_rate'][0], 2))
                print('terminal decline:', round(prod_info['terminal_decline'][0], 2))
                print('min rate:', round(prod_info['min_rate'][0], 2))
                print('prod shift:', round(prod_info['prod_shift'][0], 2))
                print('rmse:', round(prod_info['rmse'][0], 2))
                print('eur:', '{:,.0f}'.format(prod_info['eur'][0]))
                print('forecast type:', prod_info['forecast_type'][0])
                print('ratio:', round(prod_info['ratio'][0], 2))
                print('run date:', pd.Timestamp(prod_info['run_date'][0]).strftime('%x'))
            else:
                print('\r')
                print('forecast type:', well.forecasts.forecast_type)
                print('type curve:', well.forecasts.forecast)

            data = []

            if fcst:
                if forecast.prod_date[0]:
                    idf = forecast[forecast.prod_date == last_prod_date].index[0]
                    prod_fcst = go.Scatter(x=forecast.prod_date[idf:idf+500],
                                           y=forecast[prod_type][idf:idf+500],
                                           mode='lines', name='forecast')
                    data.append(prod_fcst)
                    # plt.plot(forecast.prod_date[idf:idf+500], forecast[prod_type][idf:idf+500])
                else:
                    if p is not None:
                        x_range = pd.date_range(start=prod_start_date, periods=1000, freq='d')
                        prod_fcst = go.Scatter(x=x_range,
                                               y=forecast[prod_type][:1000],
                                               mode='lines', name='forecast')
                        data.append(prod_fcst)                 
                        # plt.plot(x_range, forecast[prod_type][:1000])
                    else:
                        prod_fcst = go.Scatter(x=forecast['time_on'][:1000],
                                               y=forecast[prod_type][:1000],
                                               mode='lines', name='forecast')
                        data.append(prod_fcst)
                        # plt.plot(forecast['time_on'][:1000], forecast[prod_type][:1000])

            if tmp_fcst_dict is not None:
                if p is None:
                    p = 0
                if tmp_fcst_dict['prod_date'][0]:
                    tmp_prod_fcst = go.Scatter(x=tmp_fcst_dict['prod_date'][p:p+500],
                                               y=tmp_fcst_dict[prod_type][p:p+500],
                                               mode='lines', name='temp forecast')
                    data.append(tmp_prod_fcst)
                    # plt.plot(tmp_fcst_dict['prod_date'][p:p+500], tmp_fcst_dict[prod_type][p:p+500])
                else:
                    tmp_prod_fcst = go.Scatter(x=tmp_fcst_dict['time_on'][p:p+500],
                                               y=tmp_fcst_dict[prod_type][p:p+500],
                                               mode='lines', name='temp forecast')
                    data.append(tmp_prod_fcst)
                    # plt.plot(tmp_fcst_dict['time_on'][p:p+500], tmp_fcst_dict[prod_type][p:p+500])                        

            if prod and p:
                if smooth:
                    if production.shape[0] <= 30:
                        window = 3
                    elif production.shape[0] > 30 and production.shape[0] <= 240:
                        window = 7
                    elif production.shape[0] > 240 and production.shape[0] <= 540:
                        window = 14
                    else:
                        window = 30
                    print('smoothing window:', window)
                    prod_smooth = production[prod_type].rolling(window, center=True).mean()
                    prod_smooth = go.Scatter(x=production.prod_date,
                                             y=prod_smooth,
                                             mode='lines', name='smoothed production',
                                             line=dict(color='black'), opacity=0.75)
                    data.append(prod_smooth)
                    # plt.plot(production.prod_date, prod_smooth,
                    #             alpha=0.75, color='black')
                prod = go.Scatter(x=production.prod_date,
                                  y=production[prod_type],
                                  mode='lines', name='production',
                                  line=dict(color='slategray'), opacity=0.75)
                data.append(prod)
                # plt.plot(production.prod_date, production[prod_type],
                #             alpha=0.75, color='slategray')
                prod = False

            if prod_info:
                textstr = '\n'.join(('idp: ' + idp,
                                     'name: ' + well.well_name,
                                     'pad: ' + well.short_pad,
                                     'pod: ' + well.pad,
                                     'prod type: ' + prod_type,
                                     'date of first production: '+ prod_start_date.strftime('%m/%d/%Y'),
                                     'date of last production: ' + last_prod_date.strftime('%m/%d/%Y'),
                                     'production length: ' + str(p) + ' days',
                                     'forecast start date: ' + fcst_start_date.strftime('%m/%d/%Y'),
                                     'b: ' + str(round(prod_info['b_factor'][0], 2)),
                                     'di: ' + str(round(prod_info['initial_decline'][0], 2)),
                                     'ip: ' + '{:,.0f}'.format(prod_info['initial_rate'][0]),
                                     'prod shift: ' + str(round(prod_info['prod_shift'][0], 2)),
                                     'dmin: ' + str(prod_info['terminal_decline'][0]),
                                     'min rate: ' + str(prod_info['min_rate'][0]),
                                     'eur: ' + '{:,.0f}'.format(prod_info['eur'][0])))
                # props = dict(boxstyle='round', facecolor='white', alpha=0.25)
                # ax = plt.gca()
                # for txt in ax.texts:
                #     txt.set_visible(False)
                # textbox = ax.text(0.85, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                #                   verticalalignment='top', bbox=props)
            # plt.ylim(1, 10000)
            # plt.yscale('log')
            # plt.tight_layout()
            layout = dict(title='Title', xaxis=dict(title='Date'), yaxis=dict(title='Production', type='log'))
            fig = dict(data=data, layout=layout)
            iplot(fig)
            print('\n')
            plot_cmd = input(' (☞ﾟヮﾟ)☞ :  ')

    return

def manual_fit(well, fcst_start_date, production,
               prod_type, params, dmin, min_rate):
    fcst_dict = {'scenario': [well.forecasts.prod_forecast_scenario] * 18250,
                 'idp': [well.idp] * 18250,
                 'forecast': [well.forecasts.forecast] * 18250,
                 'time_on': np.arange(1, 18251),
                 'prod_date': [None] * 18250,
                 'prod_cat': ['forecast'] * 18250,
                 'gas': None,
                 'oil': None,
                 'water': None,
                 'cum_gas': None,
                 'cum_oil': None,
                 'cum_water': None}

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
    np.savetxt('forecast.csv', forecast)

    for t, y in well.forecasts.yields_dict.items():
        if y is not None:
            if t == prod_type:
                well.forecasts.yields_dict[t] = None
            if t == 'gas':
                fcst_dict['gas'] = fcst_dict['oil'] * y
                fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()
            if t == 'oil':
                fcst_dict['oil'] = fcst_dict['gas'] * y / 1000
                fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()
            if t == 'water':
                fcst_dict['water'] = fcst_dict['gas'] * y / 1000
                fcst_dict['cum_water'] = fcst_dict['water'].cumsum()
    well.forecasts.yields = Well_Yields(well.forecasts, well.forecasts.yields_dict)

    prod_date = pd.date_range(production.prod_date.min(), periods=18250, freq='D')
    fcst_dict['prod_date'] = prod_date

    prod_cat = ['actual'] * len(production[:start_idx])
    forecast_cat = ['forecast'] * (len(forecast) - len(production[:start_idx]))
    prod_cat.extend(forecast_cat)
    fcst_dict['prod_cat'] = prod_cat

    time_on = time_on = np.arange(1, 18251)
    fcst_dict['time_on'] = time_on

    return fcst_dict