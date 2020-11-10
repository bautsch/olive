from .utils import *
from .utils import _dotdict
from .forecaster import Well_Forecast, Well_Yields, Well_Prod_Info
import os
import pandas as pd
import numpy as np
import sys
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import calendar
import pickle
np.seterr(over='ignore')
pd.options.display.float_format = '{:,.2f}'.format


class Framework():
    def __init__(self, branch):
        self.tree = branch.tree
        self.branch = branch
        self.name = branch.scenario.framework
        self.forecast_load = None
        self.econ_load = None
        self.forecasts = None
        self.economics = None
        self.uncertainty = None
        self.risk = None
        self.delete_all = True
        self.mc_pop = False
        self.num_simulations = None
        self.econ_dists = []
        self.aggregations = {}
        self.load_framework()
        self.load_well_data()

    def __repr__(self):
        print_dict = {}
        for k, v in self.__dict__.items():
            if k in ('name', 'mc', 'mc_pop', 'num_simulations', 'effective_date',
                     'life', 'end_date', 'production_only', 'daily_output', 'start_discount'):
                print_dict[k] = v
        print_dict['effective_date'] = print_dict['effective_date'].strftime('%x')
        print_dict['end_date'] = print_dict['end_date'].strftime('%x')
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        return pretty_print(print_df)

    def load_framework(self):
        print('\ninitializing framework\n')
        sql_load = load_framework_scenario(self)

        self.effective_date = pd.Timestamp(sql_load['effective_date'])
        self.life = sql_load['life']

        if self.life > 50:
            print('max life is 50 years, setting life to 50')
            self.life = 50

        self.end_date = end_date(self)
        self.date_range = pd.date_range(self.effective_date,
                                        self.end_date, freq='d')

        self.production_only = sql_load['production_only']
        self.daily_output = sql_load['daily_output']
        self.start_discount = sql_load['start_discount']

        self.pv_spread = sql_load['pv_spread']
        if self.pv_spread is None:
            self.pv_spread = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            print('using default pv spread', ', '.join(self.pv_spread))
            sys.stdout.flush()
        else:
            self.pv_spread = self.pv_spread.split(',')
            if len(self.pv_spread) > 10:
                print('only 10 pv spread values allowed, dropping', ', '.join(self.pv_spread[10:]))
                sys.stdout.flush()
                self.pv_spread = self.pv_spread[:10]
        print('pv spread', ', '.join(self.pv_spread), '\n')
        sys.stdout.flush()

        self.min_life = int(sql_load['min_life'])

    def load_well_data(self, properties=None):
        print('initializing forecasts')
        if self.branch.scenario.forecast is None:
            print('no forecast scenario provided')
            return

        if self.branch.forecaster is None or self.branch.forecaster.forecast_load is None:
            self.forecast_load = load_forecast_scenario(self)
            self.forecasts = _dotdict()
        else:
            self.forecast_load = self.branch.forecaster.forecast_load
            self.forecasts = _dotdict({k: None for k in self.forecast_load.idp.unique()})

        if properties is not None:
            self.forecast_load = pd.concat([self.forecast_load, load_forecast_scenario(self, properties)])

        for _, row in self.forecast_load.iterrows():
            if row['idp'] in self.forecasts.keys():
                self.forecasts[row['idp']] = self.branch.model[row['idp']].forecasts
                # self.forecasts[row['idp']].forecast = row['forecast']
                # self.forecasts[row['idp']].forecast_type = row['forecast_type']
                # self.forecasts[row['idp']].yields_dict = {'gas': row['gas_g_mpb'],
                #                                           'oil': row['oil_g_bpmm'],
                #                                           'water': row['water_g_bpmm']},
                # self.forecasts[row['idp']].yields = Well_Yields({'gas': row['gas_g_mpb'],
                #                                                  'oil': row['oil_g_bpmm'],
                #                                                  'water': row['water_g_bpmm']})
            else:
                yields = {'gas': row['gas_g_mpb'], 'oil': row['oil_g_bpmm'], 'water': row['water_g_bpmm']}
                self.forecasts[row['idp']] = Well_Forecast(idp=row['idp'],
                                                           prod_forecast_scenario=row['prod_forecast_scenario'],
                                                           forecast=row['forecast'],
                                                           forecast_type=row['forecast_type'],
                                                           yields_dict=yields)

                self.forecasts[row['idp']].yields = Well_Yields(self.forecasts[row['idp']], yields)            

        if self.branch.scenario.economics is None and not self.production_only:
            print('no economics scenario provided')
            return

        if not self.production_only:
            print('initializing economics')

            if self.econ_load is None:
                self.econ_load = load_economics_scenario(self)
                self.economics = _dotdict()
            elif properties is not None:
                self.econ_load = pd.concat([self.econ_load, load_economics_scenario(self, properties)])

            for _, row in self.econ_load.iterrows():
                econ_dict = {}
                for c in self.econ_load.columns:
                    if c in ('scenario', 'idp'):
                        continue
                    econ_dict[c] = row[c]
                self.economics[row['idp']] = Well_Econ(econ_dict)

        prop = self.branch.properties
        for idp in self.forecasts.keys():
            if idp in self.branch.model.keys():
                self.branch.model[idp].framework = self
                if not self.branch.model[idp].forecasts:
                    self.branch.model[idp].forecasts = self.forecasts[idp]
            else:
                self.branch.model[idp] = _dotdict({
                                                   'tree': self.branch.tree,
                                                   'branch': self.branch,
                                                   'idp': idp,
                                                   'well_name': prop[prop.propnum == idp].bolo_well_name.values[0],
                                                   'budget_type': prop[prop.propnum == idp].budget_type.values[0],
                                                   'pad': prop[prop.propnum == idp]['pad'].values[0],
                                                   'short_pad': prop[prop.propnum == idp].short_pad.values[0],
                                                   'area': prop[prop.propnum == idp].prospect.values[0],
                                                   'project': None,
                                                   'project_id': None,
                                                   'properties': self.branch.scenario.properties,
                                                   'schedule': None,
                                                   'schedule_inputs': None,
                                                   'framework': self,
                                                   'forecasts': self.forecasts[idp]
                                                   })

        if not self.production_only:
            for idp in self.economics.keys():
                if idp in self.branch.model.keys():
                    self.branch.model[idp].economics = self.economics[idp]
                    self.branch.model[idp].price_deck = self.branch.scenario.price_deck
                else:
                    self.branch.model[idp] = _dotdict({
                                                    'tree': self.branch.tree,
                                                    'branch': self.branch,
                                                    'idp': idp,
                                                    'well_name': prop[prop.propnum == idp].bolo_well_name.values[0],
                                                    'budget_type': prop[prop.propnum == idp].budget_type.values[0],
                                                    'pad': prop[prop.propnum == idp]['pad'].values[0],
                                                    'short_pad': prop[prop.propnum == idp].short_pad.values[0],
                                                    'area': prop[prop.propnum == idp].prospect.values[0],
                                                    'project': None,
                                                    'project_id': None,
                                                    'properties': self.branch.scenario.properties,
                                                    'schedule': None,
                                                    'schedule_inputs': None,
                                                    'framework': self,
                                                    'forecasts': None,
                                                    'economics': self.economics[idp],
                                                    'price_deck': self.branch.scenario.price_deck
                                                    })
            self.load_pricing()

    def load_pricing(self):
        print('initializing price deck')
        gas_check = True
        while gas_check:
            self.price_deck = load_price_deck(self)
            min_prices = self.price_deck[self.price_deck.prod_date == self.price_deck.prod_date.min()]
            max_prices = self.price_deck[self.price_deck.prod_date == self.price_deck.prod_date.max()]
            p = self.price_deck[(self.price_deck.prod_date >= self.effective_date) &
                                (self.price_deck.prod_date <= self.end_date)]

            temp_price_df = pd.DataFrame({'prod_date': self.date_range,
                                        'input_gas_price': np.empty(len(self.date_range)),
                                        'input_oil_price': np.empty(len(self.date_range)),
                                        'input_ngl_price': np.empty(len(self.date_range))
                                        })

            temp_price_df.loc[(temp_price_df.prod_date >= p.prod_date.min()) &
                            (temp_price_df.prod_date <= p.prod_date.max()),
                            'input_gas_price'] = p.gas_price.values
            temp_price_df.loc[(temp_price_df.prod_date >= p.prod_date.min()) &
                            (temp_price_df.prod_date <= p.prod_date.max()),
                            'input_oil_price'] = p.oil_price.values
            temp_price_df.loc[(temp_price_df.prod_date >= p.prod_date.min()) &
                            (temp_price_df.prod_date <= p.prod_date.max()),
                            'input_ngl_price'] = p.ngl_price.values

            temp_price_df.loc[(temp_price_df.input_gas_price < 0.01) &
                            (temp_price_df.prod_date < min_prices.prod_date.values[0]),
                            'input_gas_price'] = min_prices.gas_price.values[0]
            temp_price_df.loc[(temp_price_df.input_oil_price < 0.01) &
                            (temp_price_df.prod_date < min_prices.prod_date.values[0]),
                            'input_oil_price'] = min_prices.oil_price.values[0]
            temp_price_df.loc[(temp_price_df.input_ngl_price < 0.01) &
                            (temp_price_df.prod_date < min_prices.prod_date.values[0]),
                            'input_ngl_price'] = min_prices.ngl_price.values[0]

            temp_price_df.loc[(temp_price_df.input_gas_price < 0.01) &
                            (temp_price_df.prod_date > max_prices.prod_date.values[0]),
                            'input_gas_price'] = max_prices.gas_price.values[0]
            temp_price_df.loc[(temp_price_df.input_oil_price < 0.01) &
                            (temp_price_df.prod_date > max_prices.prod_date.values[0]),
                            'input_oil_price'] = max_prices.oil_price.values[0]
            temp_price_df.loc[(temp_price_df.input_ngl_price < 0.01) &
                            (temp_price_df.prod_date > max_prices.prod_date.values[0]),
                            'input_ngl_price'] = max_prices.ngl_price.values[0]
            self.price_deck = temp_price_df
            if any(self.price_deck.input_gas_price > 5):
                print('gas price failed verification reloading')
            else:
                print('price deck verification succeed')
                gas_check = False

    def run_mc(self):
        start = time.time()
        print('running', self.num_simulations, 'simulations')
        sys.stdout.flush()
        idx = 0
        results = []
        start = time.time()
        property_list = self.branch.properties.propnum.values
        num_processes = int(len(property_list) / 50)
        simulations_mc = False
        if num_processes <= 4:
            if self.num_simulations < 8:
                num_processes = 1
            elif self.num_simulations < 16:
                num_processes = 2
            elif self.num_simulations < 32:
                num_processes = 4
            elif self.num_simulations < 64:
                num_processes = 6
            else:
                num_processes = 7
            chunk_size = int(self.num_simulations / num_processes)
            chunks = [[chunk_size, property_list]]*num_processes
            if chunk_size * num_processes < self.num_simulations:
                chunks.append([self.num_simulations - chunk_size * num_processes, property_list])
            if chunk_size * num_processes > self.num_simulations:
                chunks[-1][0] = chunks[-1][0] - chunk_size * num_processes - self.num_simulations
            pool = mp.Pool(processes=len(chunks))
            results = pool.map(self.run_simulations_mc, chunks)
        elif num_processes > 4:
            if num_processes > 7:
                num_processes = 7
            chunk_size = int(len(property_list) / num_processes)
            chunks = [property_list[i:i + chunk_size] for i in range(0, len(property_list), chunk_size)]
            for sim in range(self.num_simulations):
                print('simulation', sim + 1, 'of', self.num_simulations)
                sys.stdout.flush()
                results.append(self.run_populate())
                stop = time.time()
                timer(start, stop)
        print('simulations finished')
        sys.stdout.flush()
        stop = time.time()
        timer(start, stop)
        self.econ_dists = results
        return

    def run_simulations_mc(self, chunk):
        num_simulations = chunk[0]
        property_list = chunk[1]
        results = []
        for sim in range(num_simulations):
            print('simulation', sim + 1, 'of', num_simulations)
            sys.stdout.flush()
            results.append(self.prepopulate(property_list))
        for i, sim in enumerate(results):
            if i == 0:
                continue
            for k in sim.keys():
                results[0][k] = np.concatenate([results[0][k], sim[k]])
        return results[0]

    def run_populate(self):
        if not self.mc_pop:
            delete_output(self)
        print('\npopulating')
        sys.stdout.flush()
        start = time.time()
        property_list = self.branch.properties.propnum.values
        num_processes = int(len(property_list) / 50)
        if num_processes < 1:
            num_processes = 1
        if num_processes > 7:
            num_processes = 7
        chunk_size = int(len(property_list) / num_processes)
        chunks = [property_list[i:i + chunk_size] for i in range(0, len(property_list), chunk_size)]
        n = 0
        for i, chunk in enumerate(chunks):
            n += len(chunk)
            print(i+1, len(chunk))
            sys.stdout.flush()
        print(n, 'total properties in chunks', len(property_list), 'total properties in model')
        print('\n')
        sys.stdout.flush()
        pool = mp.Pool(processes=len(chunks))
        results = pool.map(self.prepopulate, chunks)
        pool.close()
        stop = time.time()
        timer(start, stop)

        if not self.mc_pop:
            self.output = results
            return
        else:
            for i, sim in enumerate(results):
                if i == 0:
                    continue
                for k in sim.keys():
                    results[0][k] = np.concatenate([results[0][k], sim[k]])
            return results[0]

    def prepopulate(self, property_list):
        start = time.time()
        if len(property_list) > 200:
            temp_c = []
            chunks = [property_list[i:i + 400] for i in range(0, len(property_list), 400)]
            for i, c in enumerate(chunks):
                print('loop', i+1, 'of', len(chunks), len(c))
                temp_c.append(self.populate(c))
                stop = time.time()
                timer(start, stop)
            return join_dict(temp_c)
        else:
            return self.populate(property_list)

    def populate(self, property_list):
        sys.stdout.flush()
        start = time.time()
        
        if self.production_only:
            #print('building gross production profile')
            sys.stdout.flush()
        else:
            #print('building gross production profile and generating economics')
            sys.stdout.flush()

        num_properties = len(property_list)
        num_days = len(self.date_range)

        columns=['scenario', 'idp', 'prod_date', 'budget_type',
                 'name', 'short_pad', 'pad',
                 'rig', 'area', 'time_on',
                 'gross_gas_mult', 'gross_oil_mult', 'gross_water_mult',
                 'gross_gas', 'gross_oil', 'gross_water',
                 'ngl_yield', 'btu', 'shrink', 'wi', 'nri', 'royalty',
                 'net_gas', 'net_oil', 'net_ngl', 'net_mcfe',
                 'royalty_gas', 'royalty_oil', 'royalty_ngl',
                 'input_gas_price', 'input_oil_price',
                 'input_ngl_price', 'gas_price_adj', 'gas_adj_unit',
                 'oil_price_adj', 'oil_adj_unit',
                 'ngl_price_adj', 'ngl_adj_unit', 'realized_gas_price',
                 'realized_oil_price', 'realized_ngl_price',
                 'net_gas_rev', 'net_oil_rev',
                 'net_ngl_rev', 'net_total_rev',
                 'gross_drill_capex', 'gross_compl_capex',
                 'gross_misc_capex', 'gross_aban_capex', 'gross_total_capex',
                 'net_drill_capex', 'net_compl_capex', 'net_misc_capex', 
                 'net_aban_capex', 'net_total_capex',
                 'fixed_cost', 'alloc_fixed_cost', 'var_gas_cost', 'var_oil_cost',
                 'var_water_cost', 'doe', 'gtp', 'tax_rate',
                 'taxes', 'loe', 'cf', 'fcf', 'cum_fcf', 'pv1', 'pv1_rate',
                 'pv2', 'pv2_rate', 'pv3', 'pv3_rate', 'pv4', 'pv4_rate',
                 'pv5', 'pv5_rate', 'pv6', 'pv6_rate', 'pv7', 'pv7_rate',
                 'pv8', 'pv8_rate', 'pv9', 'pv9_rate', 'pv10', 'pv10_rate',
                 'created_by', 'created_on']

        df = {}
        for c in columns:
            if c in ('scenario', 'idp', 'name', 'budget_type',
                     'short_pad', 'pad', 'rig', 'area'):
                df[c] = np.empty(num_properties*num_days, dtype='object')
            elif c in ('gas_adj_unit', 'oil_adj_unit', 'ngl_adj_unit'):
                df[c] = np.empty(num_properties*num_days, dtype='object')
                df[c][:] = 'none'
            elif c == 'prod_date':
                df[c] = np.empty(num_properties*num_days, dtype='datetime64[s]')
                df[c][:] = np.nan
            elif c == 'created_by':
                df[c] = [os.getlogin()] * (num_properties*num_days)
            elif c == 'created_on':
                df[c] = [pd.Timestamp(datetime.now())] * (num_properties*num_days)
            else:
                df[c] = np.zeros(num_properties*num_days, dtype='float')
        
        risk_uncertainty = {}

        for n, p in enumerate(property_list):
            idx = n * num_days
            # print('production', p, n+1, 'of', num_properties)
            sys.stdout.flush()
            f = self.forecasts[p]
            budget_flag = self.branch.model[p].budget_type
            
            if self.branch.model[p].schedule.schedule == 'base':
                budget_type = 'base'
            else:
                budget_type = 'wedge'

            risk_uncertainty[p] = {}   

            tc_mult = 1.0
            ip_mult = 1.0
            drill_mult = 1.0
            complete_mult = 1.0
            gas_price_mult = 1.0
            oil_price_mult = 1.0
            ngl_price_mult = 1.0
            infra_cost = 0.0
            curtailment = None
            frac_hit = None
            spacing = None
            drill_risk = None
            complete_risk = None
            abandon = None
            downtime = None
            gas_downtime = None
            oil_downtime = None
            delay = None

            if self.uncertainty is not None:
                u = self.uncertainty
                if 'performance' in u.keys():
                    tc_mult = apply_uncertainty(u['performance'])
                if 'profile' in u.keys():
                    ip_mult = apply_uncertainty(u['profile'])
                if 'gas_price' in u.keys():
                    gas_price_mult = apply_uncertainty(u['gas_price'])
                if 'oil_price' in u.keys():
                    oil_price_mult = apply_uncertainty(u['gas_price'])
                if 'ngl_price' in u.keys():
                    ngl_price_mult = apply_uncertainty(u['gas_price'])

                if budget_type == 'wedge':
                        if 'drill_cost' in u.keys():
                            drill_mult = apply_uncertainty(u['drill_cost'])
                        if 'complete_cost' in u.keys():
                            complete_mult = apply_uncertainty(u['complete_cost'])
                        if 'infrastructure_cost' in u.keys():
                            infra_cost = apply_uncertainty(u['infrastructure_cost'])

            if self.risk is not None:
                r = self.risk
                if budget_type == 'wedge':
                    if 'drill_cost' in r.keys():
                        drill_risk = apply_risk(r['drill_cost'])
                        if drill_risk:
                            drill_mult = drill_risk
                    if 'complete_cost' in r.keys():
                        complete_risk = apply_risk(r['complete_cost'])
                        if complete_risk:
                            complete_mult = complete_risk
                    if 'delay' in r.keys():
                        delay = apply_risk(r['delay'])
                        if delay:
                            delay = int(delay)
                    if 'abandon' in r.keys():
                        abandon = apply_risk(r['abandon'])
                        if abandon:
                            tc_mult = 0.0
                            ip_mult = None
                    if 'curtailment' in r.keys() and not abandon:
                        curtailment = apply_risk(r['curtailment'])
                    if 'frac_hit' in r.keys() and not abandon:
                        frac_hit = apply_risk(r['frac_hit'])
                    if 'spacing' in r.keys() and not abandon:
                        spacing = apply_risk(r['spacing'])                      
                    if 'downtime' in r.keys() and not abandon:
                        downtime = True
                        frequency = r['downtime']['frequency']
                        duration = r['downtime']['duration']
                        downtime_mult = r['downtime']['mult']

            if budget_type == 'wedge':
                d = self.branch.schedule.schedule_dates
                prod_start_date = pd.Timestamp(d[d.idp == p].prod_start_date.values[0])

                if delay:
                    prod_start_date += relativedelta(days=delay)

                if prod_start_date >= self.end_date:
                    continue
                if prod_start_date < self.effective_date:
                    t_start = (self.effective_date - prod_start_date).days
                    prod_start_date = self.effective_date
                elif prod_start_date >= self.effective_date:
                    t_start = 0
                t_end = (self.end_date - prod_start_date).days + 5
                rig = [self.branch.model[p].schedule.pad.rig.rig_name]*num_days

            if budget_type == 'base':
                prod_start_date = self.effective_date
                rig = ['base']*num_days
            
            if budget_type == 'wedge':
                forecast = load_forecast(self, p, f.prod_forecast_scenario, t_start, t_end)
                if forecast.empty:
                    print('missing wedge forecast', p, f.prod_forecast_scenario, f.forecast)
                    df['scenario'][idx:idx+num_days] = np.nan
                    return

            if budget_type == 'base':
                forecast = load_forecast(self, p, f.prod_forecast_scenario, t_start=None, t_end=None,
                                         eff_date=prod_start_date.strftime('%m/%d/%Y'),
                                         end_date=self.end_date.strftime('%m/%d/%Y'))
                min_date = pd.Timestamp(forecast.prod_date.min())
                if min_date > prod_start_date:
                    date_delta = (min_date - self.effective_date).days
                    padding = padding_df(forecast, date_delta)
                    padding.loc[:, 'scenario'] = f.prod_forecast_scenario
                    padding.loc[:, 'idp'] = p
                    padding.loc[:, 'forecast'] = forecast.forecast.unique()[0]
                    padding.loc[:, 'prod_date'] = pd.date_range(start=self.effective_date, periods=date_delta)
                    forecast = pd.concat([padding, forecast])
                time_on = forecast.time_on.values
                if forecast.empty:
                    print('missing base forecast', p, f.prod_forecast_scenario, f.forecast)
                    sys.stdout.flush()
                    df['scenario'][idx:idx+num_days] = np.nan
                    continue

            df['scenario'][idx:idx+num_days] = [self.branch.scenario.scenario] * num_days
            df['idp'][idx:idx+num_days] = [p] * num_days
            df['budget_type'][idx:idx+num_days] = [budget_flag] * num_days
            df['prod_date'][idx:idx+num_days] = self.date_range.values
            df['name'][idx:idx+num_days] = self.branch.model[p].well_name
            df['short_pad'][idx:idx+num_days] = self.branch.model[p].short_pad
            df['pad'][idx:idx+num_days] = self.branch.model[p].pad
            df['area'][idx:idx+num_days] = self.branch.model[p].area
            df['rig'][idx:idx+num_days] = rig

            if prod_start_date > self.effective_date:
                pad_df = padding_df(forecast, (prod_start_date - self.effective_date).days)
                time_on = np.concatenate([np.zeros(len(pad_df)), np.arange(1, len(forecast) + 1)])
                forecast = pd.concat([pad_df, forecast])
            elif budget_type == 'wedge':
                time_on = np.arange(t_start + 1, len(forecast) + 1)

            if forecast.shape[0] < num_days:
                pad_df = padding_df(forecast, num_days - forecast.shape[0])
                time_on = np.concatenate([np.arange(1, len(forecast) + 1), np.zeros(len(pad_df))])
                forecast = pd.concat([forecast, pad_df])

            if forecast.shape[0] > num_days:
                forecast = forecast.iloc[:num_days]
                time_on = time_on[:num_days]              

            df['gross_gas'][idx:idx+num_days] = forecast.gas.values * tc_mult
            df['gross_oil'][idx:idx+num_days] = forecast.oil.values * tc_mult
            df['gross_water'][idx:idx+num_days] = forecast.water.values
            df['time_on'][idx:idx+num_days] = time_on

            if ip_mult and ip_mult < 1.0:
                df['gross_gas'][idx:idx+num_days] = apply_ip_adjust(ip_mult, df['gross_gas'][idx:idx+num_days])            
                df['gross_oil'][idx:idx+num_days] = apply_ip_adjust(ip_mult, df['gross_oil'][idx:idx+num_days])
            
            if curtailment and curtailment < 1.0:
                df['gross_gas'][idx:idx+num_days] = apply_curtailment(curtailment, df['gross_gas'][idx:idx+num_days])            
                df['gross_oil'][idx:idx+num_days] = apply_curtailment(curtailment, df['gross_oil'][idx:idx+num_days])

            if spacing and spacing < 1.0:
                df['gross_gas'][idx:idx+num_days] = apply_curtailment(spacing, df['gross_gas'][idx:idx+num_days])            
                df['gross_oil'][idx:idx+num_days] = apply_curtailment(spacing, df['gross_oil'][idx:idx+num_days])

            if frac_hit:
                df['gross_gas'][idx:idx+num_days] = df['gross_gas'][idx:idx+num_days] * frac_hit    
                df['gross_oil'][idx:idx+num_days] = df['gross_oil'][idx:idx+num_days] * frac_hit

            if downtime:
                mask = event_list(frequency, duration, time_on)
                df['gross_gas'][idx:idx+num_days][mask] = df['gross_gas'][idx:idx+num_days][mask] * downtime_mult
                df['gross_oil'][idx:idx+num_days][mask] = df['gross_oil'][idx:idx+num_days][mask] * downtime_mult
                gas_downtime = sum(df['gross_gas'][idx:idx+num_days][mask] * (1 - downtime_mult))
                oil_downtime = sum(df['gross_oil'][idx:idx+num_days][mask] * (1 - downtime_mult))

            if not self.production_only:
                e = self.economics[p]
                df['input_gas_price'][idx:idx+num_days] = self.price_deck.input_gas_price.values * gas_price_mult
                df['input_oil_price'][idx:idx+num_days] = self.price_deck.input_oil_price.values * oil_price_mult
                df['input_ngl_price'][idx:idx+num_days] = self.price_deck.input_ngl_price.values * ngl_price_mult

                inputs = {}
                for i, val in e.__dict__.items():
                    if i == 'inv_g_misc':
                        inputs[i] = misc_capex_parser(val, self.effective_date, self.end_date)
                    elif i in ('inv_g_drill', 'inv_g_compl', 'inv_g_aban'):
                        continue
                    elif i in ('wi_frac', 'nri_frac', 'roy_frac', 'gross_gas_mult', 'gross_oil_mult', 'gross_water_mult'):
                        inputs[i] = econ_parser(i, val, self.effective_date,
                                                self.effective_date, self.end_date)
                    else:
                        if budget_type == 'wedge':
                            if i in ('ngl_g_bpmm', 'cost_gtp', 'price_adj_gas', 'price_adj_oil',
                                     'price_adj_ngl', 'cost_fixed', 'cost_fixed_alloc'):
                                inputs[i] = econ_parser(i, val, self.effective_date,
                                                        self.branch.model[p].schedule.prod_start_date, self.end_date)
                            else:
                                inputs[i] = econ_parser(i, val, self.effective_date,
                                                        self.branch.model[p].schedule.drill_start_date,
                                                        self.end_date)
                        if budget_type == 'base':
                            inputs[i] = econ_parser(i, val, self.effective_date,
                                                    self.effective_date, self.end_date)

                df['gross_gas_mult'][idx:idx+num_days] = inputs['gross_gas_mult'].gross_gas_mult.values
                df['gross_oil_mult'][idx:idx+num_days] = inputs['gross_oil_mult'].gross_oil_mult.values
                df['gross_water_mult'][idx:idx+num_days] = inputs['gross_water_mult'].gross_water_mult.values                

                df['gross_gas'][idx:idx+num_days] = df['gross_gas'][idx:idx+num_days] * df['gross_gas_mult'][idx:idx+num_days]
                df['gross_oil'][idx:idx+num_days] = df['gross_oil'][idx:idx+num_days] * df['gross_oil_mult'][idx:idx+num_days]
                df['gross_water'][idx:idx+num_days] = df['gross_water'][idx:idx+num_days] * df['gross_water_mult'][idx:idx+num_days]

                df['ngl_yield'][idx:idx+num_days] = inputs['ngl_g_bpmm'].ngl_g_bpmm.values
                df['btu'][idx:idx+num_days] = inputs['btu_factor'].btu_factor.values
                df['shrink'][idx:idx+num_days] = inputs['shrink_factor'].shrink_factor.values
                
                df['wi'][idx:idx+num_days] = inputs['wi_frac'].wi_frac.values
                df['nri'][idx:idx+num_days] = inputs['nri_frac'].nri_frac.values
                df['royalty'][idx:idx+num_days] = inputs['roy_frac'].roy_frac.values
              
                df['net_gas'][idx:idx+num_days] = (df['gross_gas'][idx:idx+num_days] * df['nri'][idx:idx+num_days] *
                                                   df['shrink'][idx:idx+num_days])
                df['net_oil'][idx:idx+num_days] = df['gross_oil'][idx:idx+num_days] * df['nri'][idx:idx+num_days]
                df['net_ngl'][idx:idx+num_days] = (df['gross_gas'][idx:idx+num_days] * df['ngl_yield'][idx:idx+num_days] *
                                                   df['nri'][idx:idx+num_days] / 1000)
                df['net_mcfe'][idx:idx+num_days] = (df['net_gas'][idx:idx+num_days] + df['net_oil'][idx:idx+num_days] * 6 +
                                                    df['net_ngl'][idx:idx+num_days] * 6)
                df['royalty_gas'][idx:idx+num_days] = (df['gross_gas'][idx:idx+num_days] * df['royalty'][idx:idx+num_days] *
                                                       df['shrink'][idx:idx+num_days])
                df['royalty_oil'][idx:idx+num_days] = df['gross_oil'][idx:idx+num_days] * df['royalty'][idx:idx+num_days]
                df['royalty_ngl'][idx:idx+num_days] = (df['gross_gas'][idx:idx+num_days] * df['ngl_yield'][idx:idx+num_days] *
                                                       df['royalty'][idx:idx+num_days] / 1000)

                df['gas_price_adj'][idx:idx+num_days] = inputs['price_adj_gas'].price_adj_gas.values
                df['gas_adj_unit'][idx:idx+num_days] = inputs['price_adj_gas'].unit.values

                df['oil_price_adj'][idx:idx+num_days] = inputs['price_adj_oil'].price_adj_oil.values
                df['oil_adj_unit'][idx:idx+num_days] = inputs['price_adj_oil'].unit.values

                df['ngl_price_adj'][idx:idx+num_days] = inputs['price_adj_ngl'].price_adj_ngl.values
                df['ngl_adj_unit'][idx:idx+num_days] = inputs['price_adj_ngl'].unit.values

                df['gross_misc_capex'][idx:idx+num_days] = inputs['inv_g_misc'].inv_g_misc.values

                # idp_mask = (df['idp'] == p)
                # gas_pct_adj = (df['gas_adj_unit'] == 'pct')
                # gas_per_adj = (df['gas_adj_unit'] == 'per')
                # gas_pct_mask = np.logical_and(idp_mask, gas_pct_adj)
                # gas_per_adj = np.logical_and(idp_mask, gas_per_adj)
                # print(df['realized_gas_price'][gas_pct_mask])
                # sys.stdout.flush()

                if budget_type == 'wedge':

                    drill_start_date = self.branch.model[p].schedule.drill_start_date.date()
                    drill_end_date = drill_start_date + relativedelta(days=round(self.branch.model[p].schedule.drill_time, 0) - 1)
                    alloc_drill_capex = (e.inv_g_drill * drill_mult) / round(self.branch.model[p].schedule.drill_time, 0)
                    df['gross_drill_capex'][(df['idp'] == p) &
                                            (df['prod_date'] >= np.datetime64(drill_start_date)) &
                                            (df['prod_date'] <= np.datetime64(drill_end_date))] = alloc_drill_capex

                    tmp_misc = df['gross_misc_capex'][(df['idp'] == p)
                                                       & (df['prod_date'] == np.datetime64(drill_start_date))]

                    df['gross_misc_capex'][(df['idp'] == p)
                                           & (df['prod_date'] == np.datetime64(drill_start_date))] = tmp_misc + infra_cost

                    compl_start_date = self.branch.model[p].schedule.compl_start_date.date()

                    if delay:
                        compl_start_date += relativedelta(days=delay)

                    compl_end_date = compl_start_date + relativedelta(days=round(self.branch.model[p].schedule.compl_time, 0) - 1)

                    if abandon:
                        alloc_compl_capex = abandon / round(self.branch.model[p].schedule.compl_time, 0)
                    else:
                        alloc_compl_capex = (e.inv_g_compl * complete_mult) / round(self.branch.model[p].schedule.compl_time, 0)

                    df['gross_compl_capex'][(df['idp'] == p) &
                                            (df['prod_date'] >= np.datetime64(compl_start_date)) &
                                            (df['prod_date'] <= np.datetime64(compl_end_date))] = alloc_compl_capex

                df['fixed_cost'][idx:idx+num_days] = (inputs['cost_fixed'].cost_fixed.values /
                                                      inputs['cost_fixed'].eomonth.dt.day.values) * df['wi'][idx:idx+num_days]
                df['alloc_fixed_cost'][idx:idx+num_days] = (inputs['cost_fixed_alloc'].cost_fixed_alloc.values /
                                                      inputs['cost_fixed_alloc'].eomonth.dt.day.values) * df['wi'][idx:idx+num_days]                                                    
                df['var_gas_cost'][idx:idx+num_days] = (inputs['cost_vargas'].cost_vargas.values * df['shrink'][idx:idx+num_days] *
                                                        df['gross_gas'][idx:idx+num_days]) * df['wi'][idx:idx+num_days]
                df['var_oil_cost'][idx:idx+num_days] = (inputs['cost_varoil'].cost_varoil.values *
                                                        df['gross_oil'][idx:idx+num_days]) * df['wi'][idx:idx+num_days]
                df['var_water_cost'][idx:idx+num_days] = (inputs['cost_varwater'].cost_varwater.values *
                                                          df['gross_water'][idx:idx+num_days]) * df['wi'][idx:idx+num_days]

                df['gtp'][idx:idx+num_days] = inputs['cost_gtp'].cost_gtp.values * df['net_gas'][idx:idx+num_days] * df['wi'][idx:idx+num_days]
                df['tax_rate'][idx:idx+num_days] = (inputs['tax_sev'].tax_sev.values +
                                                    inputs['tax_adval'].tax_adval.values)

            if self.mc_pop:
                risk_uncertainty[p]['tc_mult'] = tc_mult
                risk_uncertainty[p]['ip_mult'] = ip_mult
                risk_uncertainty[p]['curtailment'] = curtailment
                risk_uncertainty[p]['frac_hit'] = frac_hit
                risk_uncertainty[p]['spacing'] = spacing
                risk_uncertainty[p]['drill_mult'] = drill_mult
                risk_uncertainty[p]['complete_mult'] = complete_mult
                risk_uncertainty[p]['gas_price_mult'] = gas_price_mult
                risk_uncertainty[p]['oil_price_mult'] = oil_price_mult
                risk_uncertainty[p]['ngl_price_mult'] = ngl_price_mult
                risk_uncertainty[p]['infra_cost'] = infra_cost
                risk_uncertainty[p]['drill_risk'] = drill_risk
                risk_uncertainty[p]['complete_risk'] = complete_risk
                risk_uncertainty[p]['abandon'] = abandon
                if downtime:
                    risk_uncertainty[p]['downtime'] = sum(mask)
                    risk_uncertainty[p]['gas_downtime'] = gas_downtime
                    risk_uncertainty[p]['oil_downtime'] = oil_downtime
                else:
                    risk_uncertainty[p]['downtime'] = downtime
                    risk_uncertainty[p]['gas_downtime'] = 0
                    risk_uncertainty[p]['oil_downtime'] = 0
                risk_uncertainty[p]['delay'] = delay

            t = df['input_gas_price'][idx:idx+num_days]
            if any(t > 5):
                print('1 BAD!!!!!!!!!!!!!!!!!', p)
                sys.stdout.flush()

        if not self.production_only:

            df['realized_gas_price'][df['gas_adj_unit'] == 'pct'] = (df['input_gas_price'][df['gas_adj_unit'] == 'pct'] *
                                                                     df['gas_price_adj'][df['gas_adj_unit'] == 'pct'])

            df['realized_gas_price'][df['gas_adj_unit'] == 'per'] = (df['input_gas_price'][df['gas_adj_unit'] == 'per'] +
                                                                     df['gas_price_adj'][df['gas_adj_unit'] == 'per'])

            df['realized_gas_price'] = df['realized_gas_price'] * df['btu']

            df['realized_oil_price'][df['oil_adj_unit'] == 'pct'] = (df['input_oil_price'][df['oil_adj_unit'] == 'pct'] *
                                                                     df['oil_price_adj'][df['oil_adj_unit'] == 'pct'])
            df['realized_oil_price'][df['oil_adj_unit'] == 'per'] = (df['input_oil_price'][df['oil_adj_unit'] == 'per'] +
                                                                     df['oil_price_adj'][df['oil_adj_unit'] == 'per'])

            df['realized_ngl_price'][df['ngl_adj_unit'] == 'pct'] = (df['input_ngl_price'][df['ngl_adj_unit'] == 'pct'] *
                                                                     df['ngl_price_adj'][df['ngl_adj_unit'] == 'pct'])
            df['realized_ngl_price'][df['ngl_adj_unit'] == 'per'] = (df['input_ngl_price'][df['ngl_adj_unit'] == 'per'] +
                                                                     df['ngl_price_adj'][df['ngl_adj_unit'] == 'per'])      

            df['net_gas_rev'] = (df['net_gas'] + df['royalty_gas']) * df['realized_gas_price']
            df['net_oil_rev'] = (df['net_oil'] + df['royalty_oil']) * df['realized_oil_price']
            df['net_ngl_rev'] = (df['net_ngl'] + df['royalty_ngl']) * df['realized_ngl_price']

            df['net_total_rev'] = df['net_gas_rev'] + df['net_oil_rev'] + df['net_ngl_rev']

            df['taxes'] = df['tax_rate'] * df['net_total_rev']

            df['doe'] = (df['fixed_cost'] + df['alloc_fixed_cost'] + df['var_gas_cost'] + df['var_oil_cost'] + df['var_water_cost'])
            df['loe'] = df['doe'] + df['gtp'] + df['taxes']

            df['gross_total_capex'] = df['gross_drill_capex'] + df['gross_compl_capex'] + df['gross_misc_capex']
            df['net_drill_capex'] = df['gross_drill_capex'] * df['wi']
            df['net_compl_capex'] = df['gross_compl_capex'] * df['wi']
            df['net_misc_capex'] = df['gross_misc_capex'] * df['wi']
            df['net_total_capex'] = df['net_drill_capex'] + df['net_compl_capex'] + df['net_misc_capex']
            
            df['cf'] = df['net_total_rev'] - df['loe']
            df['fcf'] = df['cf'] - df['net_total_capex']

            t = df['input_gas_price']
            if any(t > 5):
                print('2 BAD!!!!!!!!!!!!!!!!!')
                sys.stdout.flush()

            if not self.mc_pop:
                #print('discounting')
                sys.stdout.flush()
                for n, p in enumerate(property_list):
                    #print('discounting', p, n, 'of', len(property_list))
                    sys.stdout.flush()
                    fcf_mask = ((df['idp'] == p) &
                                (df['prod_date'] >= np.datetime64(prod_start_date)) &
                                (df['net_total_capex'] < 0.01))
                    neg_fcf_mask = df['fcf'] < -100
                    combined_mask = np.logical_and(fcf_mask, neg_fcf_mask)
                    if sum(combined_mask) > 1 or sum(df['fcf'][fcf_mask]) < 100:
                        first_neg = np.argmax(combined_mask == True)
                        first_neg_date = df['prod_date'][first_neg]
                        min_life_date = self.effective_date + relativedelta(days=self.min_life)
                        if first_neg_date < min_life_date:
                            end_of_life = min_life_date
                        else:
                            end_of_life = first_neg_date
                        # if p == 'M9TEG9FCFT':
                        #     np.savetxt('tes.csv', df['gross_gas'][df['idp'] == p])
                        #     print(p, end_of_life, min_date)
                        #     sys.stdout.flush()
                        idp_mask = (df['idp'] == p)
                        date_mask = (pd.to_datetime(pd.Series(df['prod_date'])) >= end_of_life)
                        combined_mask = np.logical_and(idp_mask, date_mask)
                        for k in df.keys():
                            if k in ('scenario', 'idp', 'prod_date', 'budget_type',
                                    'name', 'short_pad', 'pad', 'rig', 'area', 'time_on',
                                    'gas_price_adj', 'oil_price_adj', 'ngl_price_adj', 'created_on', 'created_by'):
                                continue
                            else:
                                df[k][combined_mask] = 0.0

                        wi_frac = self.economics[p].__dict__['wi_frac']
                        aban_val = self.economics[p].__dict__['inv_g_aban']
                        if aban_val is not None:
                            aban = aban_capex_parser(aban_val, end_of_life, self.effective_date, self.end_date)
                            wi = econ_parser('wi_frac', wi_frac, self.effective_date, self.effective_date, self.end_date)
                            if len(df['gross_aban_capex'][df['idp'] == p]) == 0:
                                print(p, 'no output data to apply abandonment')
                                sys.stdout.flush()
                            else:
                                df['gross_aban_capex'][df['idp'] == p] = aban.inv_g_aban.values
                                df['gross_total_capex'][df['idp'] == p] = (df['gross_total_capex'][df['idp'] == p] + 
                                                                        df['gross_aban_capex'][df['idp'] == p])
                                df['net_aban_capex'][df['idp'] == p] = aban.inv_g_aban.values * wi.wi_frac.values
                                df['net_total_capex'][df['idp'] == p] = (df['net_total_capex'][df['idp'] == p] + 
                                                                        df['net_aban_capex'][df['idp'] == p])
                                df['cf'][df['idp'] == p] = (df['cf'][df['idp'] == p] - 
                                                            df['net_aban_capex'][df['idp'] == p])
                                df['fcf'][df['idp'] == p] = (df['fcf'][df['idp'] == p] - 
                                                            df['net_aban_capex'][df['idp'] == p])

                    if self.start_discount == 'eff':
                        for i in range(10):
                            if i in range(len(self.pv_spread)):
                                pv = self.pv_spread[i]
                                j = i + 1
                                df['pv'+str(j)][df['idp'] == p] = npv(df['fcf'][df['idp'] == p], float(pv)/100)
                                df['pv'+str(j)+'_rate'][df['idp'] == p] = float(pv)
                            else:
                                df['pv'+str(j)][df['idp'] == p] = float(pv)
                                df['pv'+str(j)+'_rate'][df['idp'] == p] = ''
                    if self.start_discount == 'drill':
                        print('tbd, do not use, pv will not calc')
                        sys.stdout.flush()
                        # disc_val = npv(df['fcf'][(df['idp'] == p) & (df['prod_date'] >= np.datetime64(drill_start_date))], 0.1)
                        # disc_val = np.concatenate([np.zeros(num_days - len(disc_val)), disc_val])
                        # df['npv'][df['idp'] == p] = disc_val
                        # df['cum_fcf'][df['idp'] == p] = df['fcf'][df['idp'] == p].cumsum()
            else:
                #print('calculating metrics')
                sys.stdout.flush()
                econ_dists = {'idp': np.empty(num_properties, dtype='object'),
                              'gas_eur': np.zeros(num_properties),
                              'ip90': np.zeros(num_properties),
                              'drill_cost': np.zeros(num_properties),
                              'compl_cost': np.zeros(num_properties),
                              'infra_cost': np.zeros(num_properties),
                              'avg_gas_price': np.zeros(num_properties),
                              'avg_oil_price': np.zeros(num_properties),
                              'avg_ngl_price': np.zeros(num_properties),
                              'npv': np.zeros(num_properties),
                              'irr': np.zeros(num_properties),
                              'payout': np.zeros(num_properties),
                              'year_1_roic': np.zeros(num_properties),
                              'year_1_cf': np.zeros(num_properties),
                              'year_1_fcf': np.zeros(num_properties),
                              'year_2_roic': np.zeros(num_properties),
                              'year_2_cf': np.zeros(num_properties),
                              'year_2_fcf': np.zeros(num_properties),
                              'tc_mult': np.zeros(num_properties),
                              'ip_mult': np.zeros(num_properties),
                              'curtailment': np.zeros(num_properties),
                              'frac_hit': np.zeros(num_properties),
                              'spacing': np.zeros(num_properties),
                              'drill_mult': np.zeros(num_properties),
                              'complete_mult': np.zeros(num_properties),
                              'gas_price_mult': np.zeros(num_properties),
                              'oil_price_mult': np.zeros(num_properties),
                              'ngl_price_mult': np.zeros(num_properties),
                              'drill_risk': np.zeros(num_properties),
                              'complete_risk': np.zeros(num_properties),
                              'abandon': np.zeros(num_properties),
                              'downtime': np.zeros(num_properties),
                              'gas_downtime': np.zeros(num_properties),
                              'oil_downtime': np.zeros(num_properties),
                              'delay': np.zeros(num_properties),
                              }
                for n, p in enumerate(property_list):
                    #print('economic metrics', p, n, 'of', len(property_list))
                    sys.stdout.flush()
                    df['fcf'][(df['idp'] == p) &
                              (df['prod_date'] >= np.datetime64(prod_start_date)) &
                              (df['net_total_capex'] < 0.01) &
                              (df['fcf'] < 0.0)] = 0.0
                    if self.start_discount == 'eff':
                        econ_dists['npv'][n] = npv(df['fcf'][df['idp'] == p], 0.1).sum()
                    if self.start_discount == 'drill':
                        disc_val = np.npv(df['fcf'][(df['idp'] == p) & (df['prod_date'] >= np.datetime64(drill_start_date))], 0.1)
                        disc_val = np.concatenate([np.zeros(num_days - len(disc_val)), disc_val])
                        econ_dists['npv'][n] = disc_val.sum()
                    start = np.nonzero(df['fcf'][df['idp'] == p])[0][0]
                    end = np.nonzero(df['fcf'][df['idp'] == p])[0][-1]
                    econ_dists['idp'][n] = p
                    econ_dists['gas_eur'][n] = df['gross_gas'][df['idp'] == p].sum()
                    try:
                        prod_start = np.nonzero(df['gross_gas'][df['idp'] == p])[0][0]   
                        econ_dists['ip90'][n] = df['gross_gas'][df['idp'] == p][prod_start:prod_start+90].sum()
                    except:
                        econ_dists['ip90'][n] = 0.0
                    econ_dists['drill_cost'][n] = df['gross_drill_capex'][df['idp'] == p].sum()
                    econ_dists['compl_cost'][n] = df['gross_compl_capex'][df['idp'] == p].sum()
                    econ_dists['infra_cost'][n] = df['gross_misc_capex'][df['idp'] == p].sum()
                    econ_dists['avg_gas_price'][n] = df['realized_gas_price'][df['idp'] == p].mean()
                    econ_dists['avg_oil_price'][n] = df['realized_oil_price'][df['idp'] == p].mean()
                    econ_dists['avg_ngl_price'][n] = df['realized_ngl_price'][df['idp'] == p].mean()
                    if risk_uncertainty[p]['abandon'] is None:
                        econ_dists['irr'][n] = xirr(df['fcf'][df['idp'] == p][start:end])
                    else:
                        econ_dists['irr'][n] = 0.0
                    try:
                        econ_dists['payout'][n] = np.where(df['fcf'][df['idp'] == p][start:end].cumsum() >= 0)[0][0] / (365.25/12)
                    except:
                        econ_dists['payout'][n] = np.nan
                    econ_dists['year_1_cf'][n] = df['cf'][df['idp'] == p][start:start+365].sum()
                    econ_dists['year_1_fcf'][n] = df['fcf'][df['idp'] == p][start:start+365].sum()
                    econ_dists['year_1_roic'][n] = econ_dists['year_1_cf'][n] / df['net_total_capex'][df['idp'] == p].sum()
                    econ_dists['year_2_cf'][n] = df['cf'][df['idp'] == p][start+365:start+2*365].sum()
                    econ_dists['year_2_fcf'][n] = df['fcf'][df['idp'] == p][start+365:start+2*365].sum()
                    econ_dists['year_2_roic'][n] = (econ_dists['year_1_cf'][n] +
                                                    econ_dists['year_2_cf'][n]) / df['net_total_capex'][df['idp'] == p].sum()
                    econ_dists['tc_mult'][n] = risk_uncertainty[p]['tc_mult']
                    econ_dists['ip_mult'][n] = risk_uncertainty[p]['ip_mult']
                    econ_dists['curtailment'][n] = risk_uncertainty[p]['curtailment']
                    econ_dists['frac_hit'][n] = risk_uncertainty[p]['frac_hit']
                    econ_dists['spacing'][n] = risk_uncertainty[p]['spacing']
                    econ_dists['drill_mult'][n] = risk_uncertainty[p]['drill_mult']
                    econ_dists['complete_mult'][n] = risk_uncertainty[p]['complete_mult']
                    econ_dists['gas_price_mult'][n] = risk_uncertainty[p]['gas_price_mult']
                    econ_dists['oil_price_mult'][n] = risk_uncertainty[p]['oil_price_mult']
                    econ_dists['ngl_price_mult'][n] = risk_uncertainty[p]['ngl_price_mult']
                    econ_dists['drill_risk'][n] = risk_uncertainty[p]['drill_risk']
                    econ_dists['complete_risk'][n] = risk_uncertainty[p]['complete_risk']
                    econ_dists['abandon'][n] = risk_uncertainty[p]['abandon']
                    econ_dists['downtime'][n] = risk_uncertainty[p]['downtime']
                    econ_dists['gas_downtime'][n] = risk_uncertainty[p]['gas_downtime']
                    econ_dists['oil_downtime'][n] = risk_uncertainty[p]['oil_downtime']
                    econ_dists['delay'][n] = risk_uncertainty[p]['delay']

        if not self.mc_pop:
            print('chunk completed')
            t = df['input_gas_price']
            if any(t > 5):
                print('3 BAD!!!!!!!!!!!!!!!!!')
                sys.stdout.flush()
            sys.stdout.flush()
            if self.production_only:
                return df
            df['prod_date'][df['idp'] == None] = None
            sys.stdout.flush()
            self.save_output_to_sql(df)
            return

        else:
            print('chunk completed')
            sys.stdout.flush()
            return econ_dists

    def save_output_to_sql(self, df):
        if not self.daily_output:
            sys.stdout.flush()
            t = df['input_gas_price']
            if any(t > 5):
                print('4 BAD!!!!!!!!!!!!!!!!!')
                sys.stdout.flush()
            df = pd.DataFrame(df)
            df.dropna(subset=['prod_date'], inplace=True)
            t = df.input_gas_price
            if any(t > 5):
                print('5 BAD!!!!!!!!!!!!!!!!!')
                sys.stdout.flush()
            print('converting to monthly\n')
            if len(df) > 1:
                created_by = df['created_by'].unique()[0]
                created_on = df['created_on'].unique()[0]
            eomonth = []
            for d in df['prod_date']:
                day = calendar.monthrange(d.year, d.month)[1]
                eomonth.append(datetime(d.year, d.month, day))
            df.loc[:, 'prod_date'] = pd.to_datetime(pd.Series(eomonth))
            df.loc[:, ['gas_price_adj', 'oil_price_adj', 'ngl_price_adj',
                       'gas_adj_unit', 'oil_adj_unit', 'ngl_adj_unit']] = 0.0
            df.loc[df['time_on'] > 0, 'time_on'] = 1
            df = df.groupby(by=['scenario', 'idp', 'prod_date', 'budget_type',
                           'name', 'short_pad', 'pad', 'rig', 'area'], as_index=False).sum()
            df.loc[:, 'time_on'] = df['time_on'].cumsum()
            df.loc[:, 'ngl_yield'] = df['ngl_yield'] / df['prod_date'].dt.day
            df.loc[:, 'btu'] = df['btu'] / df['prod_date'].dt.day
            df.loc[:, 'shrink'] = df['shrink'] / df['prod_date'].dt.day
            df.loc[:, 'wi'] = df['wi'] / df['prod_date'].dt.day
            df.loc[:, 'nri'] = df['nri'] / df['prod_date'].dt.day
            df.loc[:, 'royalty'] = df['royalty'] / df['prod_date'].dt.day
            df.loc[:, 'gross_gas_mult'] = df['gross_gas_mult'] / df['prod_date'].dt.day
            df.loc[:, 'gross_oil_mult'] = df['gross_oil_mult'] / df['prod_date'].dt.day
            df.loc[:, 'gross_water_mult'] = df['gross_water_mult'] / df['prod_date'].dt.day
            df.loc[:, 'input_gas_price'] = df['input_gas_price'] / df['prod_date'].dt.day
            df.loc[:, 'input_oil_price'] = df['input_oil_price'] / df['prod_date'].dt.day
            df.loc[:, 'input_ngl_price'] = df['input_ngl_price'] / df['prod_date'].dt.day
            df.loc[:, 'realized_gas_price'] = df['realized_gas_price'] / df['prod_date'].dt.day
            df.loc[:, 'realized_oil_price'] = df['realized_oil_price'] / df['prod_date'].dt.day
            df.loc[:, 'realized_ngl_price'] = df['realized_ngl_price'] / df['prod_date'].dt.day
            df.loc[:, 'cum_fcf'] = df.groupby('idp')['fcf'].cumsum()
            t = df.input_gas_price
            if any(t > 5):
                print('6 BAD!!!!!!!!!!!!!!!!!')
                sys.stdout.flush()
            for i in range(10):
                if i in range(len(self.pv_spread)):
                    pv = self.pv_spread[i]
                    j = i + 1
                    df.loc[:, 'pv'+str(j)+'_rate'] = float(pv)
                else:
                    df.loc[:, 'pv'+str(j)+'_rate'] = ''
            if len(df) > 1:
                df.loc[:, 'created_by'] = created_by
                df.loc[:, 'created_on'] = created_on
            else:
                df['created_by'] = None
                df['created_on'] = None
            self.output = df
            print('saving monthly output')
            sys.stdout.flush()
            save_output(self)
        else:
            self.output = pd.DataFrame(df)
            self.output.dropna(subset=['prod_date'], inplace=True)
            print('saving daily output')
            sys.stdout.flush()
            save_output(self)

    def aggregate(self, a, num_trials, subset=None):
        start = time.time()
        label = subset
        print('aggregating economics')
        print(a, 'samples per aggregation\t', num_trials, 'trials')
        results = {'gas_eur': np.zeros(num_trials),
                   'ip90': np.zeros(num_trials),
                   'drill_cost': np.zeros(num_trials),
                   'compl_cost': np.zeros(num_trials),
                   'infra_cost': np.zeros(num_trials),
                   'npv': np.zeros(num_trials),
                   'irr': np.zeros(num_trials),
                   'payout': np.zeros(num_trials),
                   'year_1_roic': np.zeros(num_trials),
                   'year_1_cf': np.zeros(num_trials),
                   'year_1_fcf': np.zeros(num_trials),
                   'year_2_roic': np.zeros(num_trials),
                   'year_2_cf': np.zeros(num_trials),
                   'year_2_fcf': np.zeros(num_trials)}
        ed = self.econ_dists.copy()
        if subset == 'drill':
            ed = ed[ed.drill_cost > 0]
            subset = 'Drill and Complete'
        if subset == 'duc':
            ed = ed[ed.drill_cost == 0]
            subset = 'Complete'
        if not subset:
            if isinstance(a, list):
                ed_drill = ed[ed.drill_cost > 0]
                ed_duc = ed[ed.drill_cost == 0]
        for t in range(num_trials):
            if not subset:
                if isinstance(a, list):
                    idx_list = np.random.choice(ed.index, a[0])
                    idx_list = np.concatenate([idx_list, np.random.choice(ed.index, a[1])])
                else:
                    idx_list = np.random.choice(ed.index, a)
            else:
                idx_list = np.random.choice(ed.index, a)
            temp_df = ed[ed.index.isin(idx_list)]
            for c in temp_df.columns:
                if c in results.keys():
                    results[c][t] = temp_df[c].mean()
        if subset:
            self.aggregations[subset] = pd.DataFrame(results)
        else:
            self.aggregations['All'] = pd.DataFrame(results)
        save_aggregation_to_excel(self.branch, 'main', subset)
        save_aggregation_to_excel(self.branch, 'archive', subset)
        save_aggregation_to_sql(self.branch, subset, label)
        stop = time.time()
        timer(start, stop)

    def save_framework(self):
        print('saving framework pickle')
        save_object(self, self.scenario['scenario'] + '_framework')

    def plot_aggregation(self, metric, subset=None):
        if subset == 'drill':
            subset = 'Drill and Complete'
            label = 'drill_and_complete'
        if subset == 'duc':
            subset = 'Complete'
            label = 'complete'
        if subset == 'all':
            subset = 'All'
            label  = 'all'
        if not subset:
            label = None
        n = len(self.branch.framework.aggregations.keys())
        s = []
        m = []
        for k in self.branch.framework.aggregations.keys():
            if subset:
                if k == subset:
                    t = len(self.branch.framework.aggregations[k])
                    s.append([k]*t)
                    m.append(self.branch.framework.aggregations[k][metric])
            else:
                t = len(self.branch.framework.aggregations[k])
                s.append([k]*t)
                m.append(self.branch.framework.aggregations[k][metric])
        s = np.concatenate(s)
        m = np.concatenate(m)
        df = pd.DataFrame({'scenario': s, metric: m})
        build_agg_plot(self.branch, df, label)

class Well_Econ():
    def __init__(self, econ_dict):
        self.gross_gas_mult = None
        self.gross_oil_mult = None
        self.gross_water_mult = None
        self.btu_factor = None
        self.shrink_factor = None
        self.ngl_g_bpmm = None
        self.wi_frac = None
        self.nri_frac = None
        self.roy_frac = None
        self.cost_fixed = None
        self.cost_fixed_alloc = None
        self.cost_vargas = None
        self.cost_varoil = None
        self.cost_varwater = None
        self.cost_gtp = None
        self.tax_sev = None
        self.tax_adval = None
        self.inv_g_drill = None
        self.inv_g_compl = None
        self.inv_g_misc = None
        self.inv_g_aban = None
        self.price_adj_gas = None
        self.price_adj_oil = None
        self.price_adj_ngl = None
        self.parse_inputs(econ_dict)
    
    def parse_inputs(self, econ_dict):
        for k, v in econ_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v

    def __repr__(self):
        print_dict = self.__dict__.copy()
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        return pretty_print(print_df)