from .utils import *
from .utils import _dotdict
import time
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
np.seterr(divide='ignore',invalid='ignore')
pd.options.display.float_format = '{:,.2f}'.format


class Forecaster():
    def __init__(self, branch, overwrite=False, forecast_type=None):
        self.branch = branch
        self.overwrite = overwrite
        self.forecast_type = forecast_type
        print('\nloading autoforecaster parameters')
        self.auto_params = load_auto_params(self)
        self.well_dict = None
        print('loading production forecast info')
        self.prod_info = load_prod_info(self)
        if self.prod_info.empty:
            print('no production forecast info for', self.branch.scenario.forecast)
            self.prod_info = None
        self.prod_forecasts = None
        self.log = None
        self.load_parameters()

    def __repr__(self):
        return self.branch.scenario.autoforecaster

    def load_parameters(self):
        if self.branch.framework is None or self.branch.framework.forecast_load is None:
            print('loading forecasts')
            self.forecast_load = load_forecast_scenario(self)
        else:
            self.forecast_load = self.branch.framework.forecast_load

        print('initializing wells')
        start = time.time()
        for p in self.forecast_load.idp.unique():
            prop_info = self.branch.properties[self.branch.properties.propnum == p]
            fcst_info = self.forecast_load[self.forecast_load.idp == p]
            well_name = prop_info.bolo_well_name.values[0]
            pad = prop_info['pad'].values[0]
            short_pad = prop_info.short_pad.values[0]
            area = prop_info.prospect.values[0]
            prod_forecast_scenario = fcst_info.prod_forecast_scenario.values[0]
            forecast = fcst_info.forecast.values[0]
            forecast_type = fcst_info.forecast_type.values[0]
            prod_info = {'gas': None,
                         'oil': None,
                         'water': None}
            yields_dict = {'gas': None,
                      'oil': None,
                      'water': None}
            yields_dict['gas'] = fcst_info.gas_g_mpb.values[0]
            yields_dict['oil'] = fcst_info.oil_g_bpmm.values[0]
            yields_dict['water'] = fcst_info.water_g_bpmm.values[0]
            if self.well_dict is None:
                self.well_dict = _dotdict()

            if self.prod_info is not None:
                pf = self.prod_info[self.prod_info.idp == p]
                for t in pf.prod_type.unique():
                    prod_info[t] = {}
                    for c in pf.columns:
                        if c == 'prod_type':
                            continue
                        prod_info[t][c] = pf.loc[pf.prod_type == t, c].values
            

            self.well_dict[p] = Well_Forecast(p, prod_forecast_scenario, forecast, forecast_type, yields_dict,
                                              None, prod_info, Well_Prod_Info(prod_info['gas']),
                                              Well_Prod_Info(prod_info['oil']), Well_Prod_Info(prod_info['water']))

            self.well_dict[p].yields = Well_Yields(self.well_dict[p], yields_dict)

            if p in self.branch.model.keys():
                self.branch.model[p].forecasts = self.well_dict[p]            
                self.branch.model[p].autoforecaster = self
            else:
                self.branch.model[p] = _dotdict({
                                                 'tree': self.branch.tree,
                                                 'branch': self.branch,
                                                 'idp': p,
                                                 'well_name': prop_info[prop_info.propnum == p].bolo_well_name.values[0],
                                                 'pad': prop_info[prop_info.propnum == p]['pad'].values[0],
                                                 'short_pad': prop_info[prop_info.propnum == p].short_pad.values[0],
                                                 'area': prop_info[prop_info.propnum == p].prospect.values[0],
                                                 'project': None,
                                                 'project_id': None,
                                                 'properties': self.branch.scenario.properties,
                                                 'schedule': self.branch.schedule,
                                                 'schedule_inputs': None,
                                                 'framework': self.branch.framework,
                                                 'forecasts': self.well_dict[p],
                                                 'economics': None,
                                                 'price_deck': None,
                                                 'autoforecaster': self
                                                })      
        print(len(self.branch.properties.propnum.unique()), 'properties loaded')
        stop = time.time()
        timer(start, stop)

    def autofit(self):
        print('\nautofitting forecasts')
        sys.stdout.flush()
        start = time.time()
        property_list = self.branch.properties.propnum.values
        if not self.overwrite:
            sys.stdout.flush()
            sql_props = properties_with_forecasts(self)
            property_list = [p for p in property_list if p not in sql_props]
        if self.forecast_type:
            print('only autofitting wells with forecast type:', self.forecast_type)
            filtered_props = []
            for p in property_list:
                if 'autotype' in self.branch.model[p].forecasts.forecast_type:
                    p_type = 'autotype'
                else:
                    p_type = self.branch.model[p].forecasts.forecast_type
                if isinstance(self.forecast_type, list):
                    if p_type in self.forecast_type:
                       filtered_props.append(p)
                elif p_type == self.forecast_type:
                    filtered_props.append(p)
            property_list = filtered_props
        if len(property_list) > 0:
            num_properties = len(property_list)
            num_processes = int(num_properties / 50)
            if num_processes < 1:
                num_processes = 1
            if num_processes > 7:
                num_processes = 7
            chunk_size = int(num_properties / num_processes)
            if chunk_size > 200:
                chunk_size = 200
            chunks = [property_list[i:i + chunk_size] for i in range(0, num_properties, chunk_size)]
            if len(chunks) > 8:
                runs = [chunks[i:i + 8] for i in range(0, len(chunks), 8)]
            else:
                runs = None
            if runs:
                num_runs = len(runs)
                logs = []
                well_dict = []
                for i, chunk in enumerate(runs):
                    print(i+1, 'of', num_runs)
                    sys.stdout.flush()
                    n = 0
                    for i, subchunk in enumerate(chunk):
                        n += len(subchunk)
                        print(i+1, len(subchunk))
                        sys.stdout.flush()
                    print(n, 'total properties in chunks', num_properties, 'total properties in model')
                    sys.stdout.flush()
                    pool = mp.Pool(processes=len(chunk))
                    results = pool.map(self.get_fits, chunk)
                    pool.close()
                    logs.extend([r[0] for r in results])
                    well_dict.extend([r[1] for r in results])
                    fits = [r[2] for r in results]
                    yields = [r[3] for r in results]
                    print('preparing dataframe')
                    sys.stdout.flush()
                    if len(fits) > 1:
                        fits = [pd.DataFrame(f) for f in fits if all(v is not None for v in f.values())]
                        fits = pd.concat(fits)
                    else:
                        fits = pd.DataFrame(fits[0])
                    fits.dropna(subset=['scenario'], inplace=True)
                    fits.loc[:, ['gas', 'oil', 'water',
                                 'cum_gas', 'cum_oil', 'cum_water']] = fits.loc[:, ['gas', 'oil', 'water',
                                 'cum_gas', 'cum_oil', 'cum_water']].fillna(value=0.0)
                    yields = [pd.DataFrame(y) for y in yields if y is not None]
                    yields = pd.concat(yields)
                    print('saving forecasts')
                    sys.stdout.flush()
                    save_prod_forecasts(self, fits)
                    del fits

                self.log = logs
                self.well_dict = well_dict
         
            else:
                n = 0
                for i, chunk in enumerate(chunks):
                    n += len(chunk)
                    print(i+1, len(chunk))
                    sys.stdout.flush()
                print(n, 'total properties in chunks', num_properties, 'total properties in model')
                sys.stdout.flush()
                pool = mp.Pool(processes=len(chunks))
                results = pool.map(self.get_fits, chunks)
                pool.close()
                logs = [r[0] for r in results]
                well_dict = [r[1] for r in results]
                fits = [r[2] for r in results]
                yields = [r[3] for r in results]
                print('preparing dataframe')
                sys.stdout.flush()
                if len(fits) > 1:
                    fits = [pd.DataFrame(f) for f in fits if all(v is not None for v in f.values())]
                    fits = pd.concat(fits)
                else:
                    fits = pd.DataFrame(fits[0])
                fits.dropna(subset=['scenario'], inplace=True)
                fits.loc[:, ['gas', 'oil', 'water',
                                'cum_gas', 'cum_oil', 'cum_water']] = fits.loc[:, ['gas', 'oil', 'water',
                                'cum_gas', 'cum_oil', 'cum_water']].fillna(value=0.0)
                yields = [pd.DataFrame(y) for y in yields if y is not None]
                yields = pd.concat(yields)
                print('saving forecasts')
                sys.stdout.flush()
                save_prod_forecasts(self, fits)
                update_yields(self, yields)

                self.log = logs
                self.well_dict = well_dict

        else:
            print('no properties forecasted')

        stop = time.time()
        timer(start, stop)

        return

    def get_fits(self, property_list):
        num_properties = len(property_list)
        fits = {'scenario': None,
                'idp': None,
                'forecast': None,
                'time_on': None,
                'prod_date': None,
                'prod_cat': None,
                'gas': None,
                'oil': None,
                'water': None,
                'cum_gas': None,
                'cum_oil': None,
                'cum_water': None}

        log = {'scenario': None,
               'idp': None,
               'message': None,
               'run_date': None}
        
        yield_update = {'scenario': [],
                        'idp': [],
                        'gas_g_mpb': [],
                        'oil_g_bpmm': [],
                        'water_g_bpmm': []}

        tmp_well_dict = {}
        for i, p in enumerate(property_list):
            w = self.well_dict[p]
            w.prod_info = {'gas': None,
                           'oil': None,
                           'water': None}
            #print(i+1, 'of', num_properties, w.idp, w.forecast_type)
            sys.stdout.flush()
            
            if w.forecast_type == 'type':
                tmp_fit = self.fit_type_curve(w)
                for k in tmp_fit.keys():
                    if fits[k] is None:
                        fits[k] = tmp_fit[k]
                    else:
                        if tmp_fit[k] is None:
                            continue
                        fits[k] = np.concatenate([fits[k], tmp_fit[k]])

            if w.forecast_type == 'manual':
                continue

            if w.forecast_type == 'auto' or 'autotype' in w.forecast_type:
                w.prod_types = []
                for k, v in w.yields_dict.items():
                    if v is None:
                        w.prod_types.append(k)
                cols = ['prod_date']
                cols.extend(w.prod_types)
                production = load_production(self.branch, [p])
                w.production = production

                if production.empty:
                    print(p, 'has no production')
                    sys.stdout.flush()
                    continue
                else:
                    w.prod_info, tmp_log = self.validate(w, production.loc[:, cols])

                    if tmp_log is not None:
                        for k in tmp_log.keys():
                            if log[k] is None:
                                log[k] = [tmp_log[k]]
                            else:
                                log[k].append(tmp_log[k])
                        if log['run_date'] is None:
                            log['run_date'] = [pd.Timestamp(self.branch.tree.run_time)]
                        else:
                            log['run_date'].append(pd.Timestamp(self.branch.tree.run_time))
                            
                    if w.prod_info is not None:
                        if w.forecast_type == 'auto':

                            w.yields_dict = w.yields.calculate(production)
                            yield_update['scenario'].append(self.branch.scenario.forecast)
                            yield_update['idp'].append(p)
                            yield_update['gas_g_mpb'].append(w.yields_dict['gas'])
                            yield_update['oil_g_bpmm'].append(w.yields_dict['oil'])
                            yield_update['water_g_bpmm'].append(w.yields_dict['water'])

                            tmp_fit, tmp_log = self.fit(w)

                            for k in tmp_fit.keys():
                                if fits[k] is None:
                                    fits[k] = tmp_fit[k]
                                else:
                                    if tmp_fit[k] is None:
                                        continue
                                    fits[k] = np.concatenate([np.asarray(fits[k]), np.asarray(tmp_fit[k])])

                            if tmp_log is not None:
                                for k in tmp_log.keys():
                                    if log[k] is None:
                                        log[k] = [tmp_log[k]]
                                    else:
                                        log[k].append(tmp_log[k])
                                if log['run_date'] is None:
                                    log['run_date'] = [pd.Timestamp(self.branch.tree.run_time)]
                                else:
                                    log['run_date'].append(pd.Timestamp(self.branch.tree.run_time))

                    if 'autotype' in w.forecast_type:
                        if ':' in w.forecast_type:
                            ratios = [float(i) for i in w.forecast_type.split(':')[1:]]
                        else:
                            ratios = None
                        tmp_fit = self.auto_type_curve(w, ratios)
                        for k in tmp_fit.keys():
                            if fits[k] is None:
                                fits[k] = tmp_fit[k]
                            else:
                                if tmp_fit[k] is None:
                                    continue
                                fits[k] = np.concatenate([np.asarray(fits[k]), np.asarray(tmp_fit[k])])

            tmp_well_dict[p] = w
        
        print('chunk completed')
        sys.stdout.flush()
        return [log, tmp_well_dict, fits, yield_update]

    def validate(self, well, production):
        prod_info = {'gas': None,
                     'oil': None,
                     'water': None}
        log = {'scenario': None,
               'idp': None,
               'message': None}

        production.reset_index(drop=True, inplace=True)

        if production.shape[0] > 7:
            production = production[:-3]
        time_on = production.shape[0]

        for t in well.prod_types:
            if production[t].sum() < 1 and well.forecast_type == 'auto':
                print(well.idp, 'sum of', t, 'production history is less than 1')
                sys.stdout.flush()
                log['scenario'] = self.branch.scenario.forecast
                log['idp'] = well.idp
                log['message'] = 'production history is less than 1'
                return None, log

            if time_on < 45  and well.forecast_type == 'auto':
                print(well.idp, 'less than 45 days of production history')
                sys.stdout.flush()
                log['scenario'] = self.branch.scenario.forecast
                log['idp'] = well.idp
                log['message'] = 'less than 45 days of production history'
                return None, log

            max_idx = production.loc[:120, t].idxmax()
            if max_idx == 120  and well.forecast_type == 'auto' and time_on <= 120:
                print(well.idp, 'max_idx', t, 'rate not reached')
                sys.stdout.flush()
                log['scenario'] = self.branch.scenario.forecast
                log['idp'] = well.idp
                log['message'] = str('max_idx ' + t + ' rate not reached')
                return None, log

            if time_on - max_idx - 2 < 15  and well.forecast_type == 'auto':
                print(well.idp, 'less than 15 days after', t, 'max rate')
                sys.stdout.flush()
                log['scenario'] = self.branch.scenario.forecast
                log['idp'] = well.idp
                log['message'] = str('less than 15 days after ' + t + ' max rate')
                return None, log

            prod_time = time_on - max_idx
            max_rate = production[t].max()
            start_date = production.prod_date.min()
            if pd.isna(max_idx):
                print(well.idp, max_rate, start_date, max_idx, t)
            sys.stdout.flush()
            max_date = production.loc[max_idx, 'prod_date']
            if max_idx + 2 < time_on:
                fcst_start_date = production.loc[max_idx + 2, 'prod_date']
            else:
                fcst_start_date = production.loc[max_idx, 'prod_date']

            prod_info[t]= {'scenario': self.branch.scenario.forecast,
                           'idp': well.idp,
                           'type_curve': well.forecast,
                           'y': production[t].values,
                           'x': production.prod_date.values,
                           'y_fcst': production.loc[max_idx:, t].values,
                           'prod_type': t,
                           'time_on': time_on,
                           'max_idx': max_idx,
                           'max_rate': max_rate,
                           'max_date': max_date,
                           'fcst_start_date': fcst_start_date,
                           'fcst_prod_time': time_on,
                           'first_prod_date': start_date,
                           'forecast_type': well.forecast_type}
        if log['idp'] is None:
            log = None

        return prod_info, log

    def fit(self, well):
        fcst_dict = {'scenario': None,
                     'idp': None,
                     'forecast': None,
                     'prod_date': None,
                     'prod_cat': None,
                     'time_on': None,
                     'gas': None,
                     'oil': None,
                     'water': None,
                     'cum_gas': None,
                     'cum_oil': None,
                     'cum_water': None}
        log = {'scenario': None,
               'idp': None,
               'message': None}

        max_life = None
        max_t = None
        for t in well.prod_types:
            guess = self.auto_params[t]['initial_guesses']
            bounds = self.auto_params[t]['bounds']
            dmin = self.auto_params[t]['dmin']
            min_rate = self.auto_params[t]['min_rate']
            if well.prod_info[t]['max_rate'] < 10:
                guess[2] = 10.
            else:
                guess[2] = well.prod_info[t]['max_rate']

            if all(well.prod_info[t]['fcst_prod_time'] > b for b in bounds.keys()):
                r = least_squares(residuals,
                                guess,
                                loss='arctan',
                                bounds=(bounds[-1][0], bounds[-1][1]),
                                args=(well.prod_info[t]['y_fcst'], dmin, min_rate)
                                )
            else:
                temp_dict = {well.prod_info[t]['fcst_prod_time'] - n: b for n,
                            b in bounds.items()}
                temp_dict = {m: c for m, c in temp_dict.items() if m <= 0}

                b = max(temp_dict.keys())
                r = least_squares(residuals,
                                guess,
                                loss='arctan',
                                bounds=(temp_dict[b][0], temp_dict[b][1]),
                                args=(well.prod_info[t]['y_fcst'], dmin, min_rate)
                                )

            params = [r.x[0], r.x[1], r.x[2]]
            initial_rate = params[2]

            resids = residuals(params, well.prod_info[t]['y_fcst'],
                                    dmin, min_rate, method='frac')

            if well.prod_info[t]['fcst_prod_time'] >= 180 and well.prod_info[t]['fcst_prod_time'] < 365:
                prod_shift = np.mean(resids[-14:])
            if well.prod_info[t]['fcst_prod_time'] < 180:
                prod_shift = np.mean(resids[-14:]) * 0.98
            if well.prod_info[t]['fcst_prod_time'] >= 365:
                prod_shift = np.mean(resids[-60:])

            resids_diff = residuals(params, well.prod_info[t]['y_fcst'],
                                    dmin, min_rate, method='diff')

            rmse = np.sqrt(np.mean(resids_diff**2))

            if prod_shift > 0:
                params[2] *= prod_shift

            forecast = arps_fit(params, dmin, min_rate)
            start_idx = len(well.prod_info[t]['y'])
            max_idx = well.prod_info[t]['max_idx']
            if well.prod_info[t]['y'][-14:].sum() < 1:
                forecast = np.multiply(forecast, 0)
                print(well.idp, 'no production in last 14 days, setting forecast to zero')
                log['scenario'] = self.branch.scenario.scenario
                log['idp'] = well.idp
                log['message'] = 'no production in last 14 days, setting forecast to zero'
                sys.stdout.flush()
            forecast = np.concatenate([well.prod_info[t]['y'], forecast[start_idx:]])

            life = len(forecast)
            if max_life is None:
                max_life = life
                max_t = t
                max_y = len(well.prod_info[t]['y'])
            else:
                if life > max_life:
                    max_life = life
                    max_t = t
                    max_y = len(well.prod_info[t]['y'])
            eur = forecast.sum()

            fcst_dict[t] = forecast
            fcst_dict[str('cum_' + t)] = forecast.cumsum()

            well.prod_info[t]['b_factor'] = params[0]
            well.prod_info[t]['initial_decline'] = params[1]
            well.prod_info[t]['initial_rate'] = initial_rate
            well.prod_info[t]['terminal_decline'] = dmin
            well.prod_info[t]['min_rate'] = min_rate
            well.prod_info[t]['prod_shift'] = prod_shift
            well.prod_info[t]['rmse'] = rmse
            well.prod_info[t]['eur'] = eur
            well.prod_info[t]['ratio'] = 0.0
            well.prod_info[t]['run_date'] = pd.Timestamp(self.branch.tree.run_time)

            well.prod_info[t]['y'] = None
            well.prod_info[t]['x'] = None
            well.prod_info[t]['y_fcst'] = None

        fcst_dict['scenario'] = [self.branch.scenario.forecast] * max_life
        fcst_dict['idp'] = [well.idp] * max_life
        fcst_dict['forecast'] = [well.forecast] * max_life

        time_on = np.arange(1, max_life+1)
        fcst_dict['time_on'] = time_on
        
        prod_date = pd.date_range(well.prod_info[t]['first_prod_date'],
                                    periods=max_life, freq='D').strftime('%x')
        fcst_dict['prod_date'] = prod_date

        prod_cat = ['actual'] * max_y
        forecast_cat = ['forecast'] * (max_life - max_y)
        prod_cat.extend(forecast_cat)
        fcst_dict['prod_cat'] = prod_cat

        for t in well.prod_types:
            if fcst_dict[t] != max_t:
                forecast = np.concatenate([fcst_dict[t], np.zeros(max_life - len(fcst_dict[t]))])
                fcst_dict[t] = forecast
                fcst_dict[str('cum_' + t)] = forecast.cumsum()

        for t, y in well.yields_dict.items():
            if y is not None:
                if t == 'gas':
                    if well.production['gas'].sum() > 0.0:
                        fcst_dict['gas'] = np.concatenate([well.production['gas'][:start_idx],
                                                    fcst_dict['oil'][start_idx:] * y])
                    else:
                        fcst_dict['gas'] = fcst_dict['oil'] * y
                    fcst_dict[t] = np.nan_to_num(fcst_dict[t], copy=True, nan=0.0, neginf=0.0, posinf=0.0)
                    fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()
                if t == 'oil':
                    if well.production['oil'].sum() > 0.0:
                        fcst_dict['oil'] = np.concatenate([well.production['oil'][:start_idx],
                                                           fcst_dict['gas'][start_idx:] * y / 1000])
                    else:
                        fcst_dict['oil'] = fcst_dict['gas'] * y / 1000
                    fcst_dict[t] = np.nan_to_num(fcst_dict[t], copy=True, nan=0.0, neginf=0.0, posinf=0.0)
                    fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()
                if t == 'water':
                    if well.production['water'].sum() > 0.0:
                        fcst_dict['water'] = np.concatenate([well.production['water'][:start_idx],
                                                    fcst_dict['gas'][start_idx:] * y / 1000])
                    else:
                        fcst_dict['water'] = fcst_dict['gas'] * y / 1000
                    fcst_dict[t] = np.nan_to_num(fcst_dict[t], copy=True, nan=0.0, neginf=0.0, posinf=0.0)
                    fcst_dict['cum_water'] = fcst_dict['water'].cumsum()

        well.production = None

        if log['idp'] is None:
            log = None

        return fcst_dict, log

    def auto_type_curve(self, well, ratios=None):
        fcst_dict = {'scenario': [self.branch.scenario.forecast] * 18250,
                     'idp': [well.idp] * 18250,
                     'forecast': [well.forecast] * 18250,
                     'prod_date': None,
                     'prod_cat': None,
                     'time_on': None,
                     'gas': None,
                     'oil': None,
                     'water': None,
                     'cum_gas': None,
                     'cum_oil': None,
                     'cum_water': None}

        type_curve = load_type_curve(self, well.forecast)

        for t in well.prod_types:

            if well.prod_info[t]['x'].shape[0] > 5:
                production = well.prod_info[t]['y']
                start_idx = len(production)
            prod_start = pd.Timestamp(well.prod_info[t]['x'].min())

            type_curve_ratio = type_curve[:start_idx]

            if ratios is None:
                ratio = np.divide(production[:start_idx],
                                  type_curve_ratio).mean()

            elif isinstance(ratios, float):
                ratio = ratios
                type_curve = type_curve * ratio
            
            else:
                ratio = np.divide(production[:start_idx],
                                  type_curve_ratio[t]).mean()
                ratio = min(ratios[1], ratio)
                ratio = max(ratios[0], ratio)
                
                type_curve.loc[:, t] = type_curve.loc[:, t].multiply(ratio)

            well.prod_info[t]['ratio'] = ratio

            type_curve_forecast = type_curve[start_idx:]
            forecast = np.concatenate([production[:start_idx], type_curve_forecast[t]])
            if len(forecast) > 18250:
                forecast = forecast[:18250]
            if len(forecast) < 18250:
                forecast = np.concatenate([forecast, np.zeros(18250-len(forecast))])

            well.prod_info[t]['eur'] = forecast.sum()
            
            if fcst_dict['time_on'] is None:
                fcst_dict['time_on'] = np.arange(1, 18251)
            
            if fcst_dict['prod_date'] is None:
                fcst_dict['prod_date'] = pd.date_range(prod_start, periods=18250, freq='D').strftime('%x')

            fcst_dict[t] = forecast
            fcst_dict[str('cum_' + t)] = forecast.cumsum()

            if fcst_dict['prod_cat'] is None:
                prod_cat = ['actual'] * len(production[:start_idx])
                forecast_cat = ['forecast'] * (len(forecast) - len(production[:start_idx]))
                prod_cat.extend(forecast_cat)
                fcst_dict['prod_cat'] = prod_cat

            well.prod_info[t]['b_factor'] = 0.0
            well.prod_info[t]['initial_decline'] = 0.0
            well.prod_info[t]['initial_rate'] = 0.0
            well.prod_info[t]['terminal_decline'] = 0.0
            well.prod_info[t]['min_rate'] = 0.0
            well.prod_info[t]['prod_shift'] = 0.0
            well.prod_info[t]['rmse'] = 0.0
            well.prod_info[t]['run_date'] = pd.Timestamp(self.branch.tree.run_time)

            well.prod_info[t]['x'] = None
            well.prod_info[t]['y'] = None
            well.prod_info[t]['y_fcst'] = None   

        for t, y in well.yields_dict.items():
            if y is not None:
                sys.stdout.flush()
                if t == 'gas':
                    fcst_dict['gas'] = fcst_dict['oil'] * y
                    fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()
                if t == 'oil':
                    fcst_dict['oil'] = fcst_dict['gas'] * y / 1000
                    fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()
                if t == 'water':
                    fcst_dict['water'] = fcst_dict['gas'] * y / 1000
                    fcst_dict['cum_water'] = fcst_dict['water'].cumsum()

        if pd.isnull(fcst_dict['gas'][0]):
            fcst_dict['gas'] = type_curve.loc[:18250, 'gas'].multiply(ratio)
            fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()

        if pd.isnull(fcst_dict['oil'][0]):
            fcst_dict['oil'] = type_curve.loc[:18250, 'oil'].multiply(ratio)
            fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()

        if pd.isnull(fcst_dict['water'][0]):
            fcst_dict['water'] = type_curve.loc[:18250, 'water'].multiply(ratio)
            fcst_dict['cum_water'] = fcst_dict['water'].cumsum()

        return fcst_dict

    def fit_type_curve(self, well):
        fcst_dict = {'scenario': [self.branch.scenario.forecast] * 18250,
                     'idp': [well.idp] * 18250,
                     'forecast': [well.forecast] * 18250,
                     'prod_date': [None] * 18250,
                     'prod_cat': ['forecast'] * 18250,
                     'time_on': np.arange(1, 18251),
                     'gas': None,
                     'oil': None,
                     'water': None,
                     'cum_gas': None,
                     'cum_oil': None,
                     'cum_water': None}

        type_curve = load_type_curve(self, well.forecast)

        fcst_dict['gas'] = type_curve['gas'][:18250]
        fcst_dict['oil'] = type_curve['oil'][:18250]
        fcst_dict['water'] = type_curve['water'][:18250]
        fcst_dict['cum_gas'] = fcst_dict['gas'].cumsum()
        fcst_dict['cum_oil'] = fcst_dict['oil'].cumsum()
        fcst_dict['cum_water'] = fcst_dict['water'].cumsum()
        return fcst_dict

    def build_prod_info(self):
        num_properties = len(self.well_dict)
        prod_info = {'scenario': [],
                     'idp': [],
                     'type_curve': [],
                     'prod_type': [],
                     'time_on': [],
                     'fcst_prod_time': [],
                     'max_idx': [],
                     'max_rate': [],
                     'max_date': [],
                     'first_prod_date': [],
                     'fcst_start_date': [],
                     'b_factor': [],
                     'initial_decline': [],
                     'initial_rate': [],
                     'terminal_decline': [],
                     'min_rate': [],
                     'prod_shift': [],
                     'rmse': [],
                     'eur': [],
                     'forecast_type': [],
                     'ratio': [],
                     'run_date': []}

        for w in self.well_dict.values():
                if w.prod_info:
                    for t in w.prod_info:
                        if w.prod_info[t]:
                            self.branch.model[w.idp].forecasts[str(t + '_prod_info')] = Well_Prod_Info(w.prod_info[t])
                            for k in prod_info.keys():
                                if k not in w.prod_info[t].keys():
                                    prod_info[k].append(None)
                                else:
                                    prod_info[k].append(w.prod_info[t][k])

        return prod_info


class Well_Forecast():
    def __init__(self, idp, prod_forecast_scenario, forecast, forecast_type,
                 yields_dict=None, yields=None, prod_info=None,
                 gas_prod_info=None, oil_prod_info=None, water_prod_info=None):
        self.idp = idp
        self.prod_forecast_scenario = prod_forecast_scenario
        self.forecast = forecast
        self.forecast_type = forecast_type
        self.yields_dict = yields_dict
        self.yields = yields
        self.prod_info = prod_info
        self.gas_prod_info = gas_prod_info
        self.oil_prod_info = oil_prod_info
        self.water_prod_info = water_prod_info

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        self[k] = v

    def __setitem__(self, k, v):
        super().__setattr__(k, v)

    def __repr__(self):
        print_dict = self.__dict__.copy()
        del print_dict['idp']
        del print_dict['yields_dict']
        del print_dict['prod_info']
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        return pretty_print(print_df)


class Well_Yields():
    def __init__(self, forecast, yields_dict):
        self.forecast = forecast
        self.gas = None
        self.oil = None
        self.water = None
        self.parse_inputs(yields_dict)

    def parse_inputs(self, yields_dict):
        for k, v in yields_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v

    def calculate(self, production):
        p = production.dropna()
        p = p.tail(365)
        p = p.loc[(p !=0 ).any(axis=1)]

        if self.oil is not None:
            self.oil = max(p.oil.sum() / p.gas.sum() * 1000, 0)
        if self.gas is not None:
            self.gas = max(p.gas.sum() / p.oil.sum(), 0)
        if self.water is not None:
            self.water = max(p.water.sum() / p.gas.sum(), 0)

        del p

        return {'oil': self.oil,
                'gas': self.gas,
                'water': self.water}

    def __repr__(self):
        print_dict = self.__dict__.copy()
        del print_dict['forecast']
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        return pretty_print(print_df)


class Well_Prod_Info():
    def __init__(self, prod_dict):
        self.prod_type = None
        self.time_on = None
        self.fcst_prod_time = None
        self.max_idx = None
        self.max_rate = None
        self.max_date = None
        self.first_prod_date = None
        self.fcst_start_date = None
        self.b_factor = None
        self.initial_decline = None
        self.terminal_decline = None
        self.min_rate = None
        self.prod_shift = None
        self.rmse = None
        self.eur = None
        self.forecast_type = None
        self.ratio = None
        self.run_date = None
        self.parse_inputs(prod_dict)

    def parse_inputs(self, prod_dict):
        if isinstance(prod_dict, dict):
            for k, v in prod_dict.items():
                if k in self.__dict__.keys():
                    self.__dict__[k] = v

    def __repr__(self):
        print_dict = self.__dict__.copy()
        print_df = pd.DataFrame(print_dict, index=[0])
        for c in print_df.columns:
            if 'date' in c:
                d = pd.Timestamp(print_df.loc[:, c].values[0])
                if not pd.isnull(d):
                    print_df.loc[:, c] = d.strftime('%x')
        print_df = print_df.transpose()
        return pretty_print(print_df)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        self[k] = v

    def __setitem__(self, k, v):
        super().__setattr__(k, v)