from .utils import *
from .utils import _dotdict
from .schedule import Schedule, Well_Sched
from .framework import Framework
from .capacity import Capacity
from .forecaster import Forecaster
from .plotter import *
# from .dash_test2 import run_server
import time
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from datetime import date
import os
import sys
import subprocess
pd.options.display.float_format = '{:,.2f}'.format


def load_tree(tree_name):
    print('loading tree:\t\t\t\t'+tree_name)
    tree = load_object(tree_name + '\\' + tree_name)
    for branch in tree.branches.values():
        check_scenario(branch, branch.scenario)
    return tree


class Tree():
    def __init__(self, tree_name, create_folders=True, verbose=True,
                 connection_dict={'driver': 'ODBC Driver 17 for SQL Server',
                                  'server': 'COG-DEN-ARIES',
                                  'database': 'olive',
                                  'credentials': None}):
        if verbose:
            print('creating tree:\t\t\t\t'+tree_name)
        self.verbose = verbose
        self.name = tree_name
        self.connection_dict = connection_dict
        self.branches = {}
        if create_folders:
            self.create_folders()
        self.run_time = time.strftime('%Y%m%d-%H%M%S')

    def __repr__(self):
        return self.name

## Main methods for interacting with Olive ##
    def get_scenarios(self):
        scen = load_all_scenarios(self)

        scen_dict = {}
        for _, row in scen.iterrows():
            temp_dict = {}
            for c in scen.columns:
                if c == 'scenario':
                    continue
                temp_dict[c] = row[c]
            scen_dict[row['scenario']] = _dotdict(temp_dict)
        self.scenarios = _dotdict(scen_dict)

        print_scen = scen.transpose()
        pdtabulate = lambda df:tabulate(df, tablefmt='grid')
        print(pdtabulate(print_scen))

    def add_scenario(self, scenario_name, scenario_dict, overwrite=False):
        self.scenarios[scenario_name] = _dotdict(scenario_dict)
        scenario = {}
        scenario.scenario = scenario_name
        for k, v in scenario_dict.items():
            scenario[k] = v
        save_scenario_to_sql(self, scenario, overwrite)

    def save_scenario(self, scenario_name, overwrite=True):
        scenario = {}
        scenario.scenario = scenario_name
        for k, v in self.scenarios[scenario_name].__dict__.items():
            scenario[k] = v
        save_scenario_to_sql(self, scenario, overwrite=True)

    def update_production(self):
        update_daily_production(self)

    def load_branch(self, branch_path, branch_name=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if branch_name is None:
            branch_name = branch_path
        self.branches[branch_name] = load_object(branch_path)
        if verbose:
            print('loading branch:\t\t\t\t'+branch_name)
            check_scenario(self.branches[branch_name], self.branches[branch_name].scenario)        

    def add_branch(self, branch_name, scenario=None, max_date=None):
        self.branches[branch_name] = Branch(branch_name, self, scenario, max_date)

    def save_tree(self):
        print('saving tree to pickle')
        save_object(self, self.name + '\\' + self.name)

    def export_data(self):
        print('exporting data')
        name = self.branches[list(self.branches.keys())[0]].name
        file_path = str(self.name + '\\archive\\' + self.name + '_' +
                        name + '_export_' + self.run_time + '.xlsx')
        run_export(self, file_path)
        file_path = str(self.name + '\\' + self.name + '_' +
                        name + '_export.xlsx')
        run_export(self, file_path)

## Helper functions use primarily for multiprocessing ##
    def create_folders(self):
        create_dir(self)

    def dash_test(self, branch_name):
        branch = self.branches[branch_name]
        start = time.time()
        build_script_path = os.path.dirname(__file__)
        python_path = sys.executable
        print('preparing dash test')
        p = subprocess.Popen(python_path + ' ' + build_script_path + '\\dash_test.py',
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        for line in p.stdout:
            print(line.rstrip())
        while not os.path.exists('temp\\reload.pkl'):
            time.sleep(5)
        if os.path.isfile('temp\\reload.pkl'):
            print('\nloading temp reload file')
            self.branches['temp'] = load_object('temp\\reload')
        print('cleaning up')
        while os.path.exists('temp\\reload.pkl'):
            try:
                if os.path.isfile('temp\\reload.pkl'):
                    os.remove('temp\\reload.pkl')
            except PermissionError:
                time.sleep(3)

    def build_output(self, branch_name):
        branch = self.branches[branch_name]
        start = time.time()
        build_script_path = os.path.dirname(__file__)
        python_path = sys.executable
        print('preparing output run')
        p = subprocess.Popen(python_path + ' ' + build_script_path + '\\build_output.py',
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        for line in p.stdout:
            print(line.rstrip())
        while not os.path.exists('temp\\reload.pkl'):
            time.sleep(5)
        if os.path.isfile('temp\\reload.pkl'):
            print('\nloading temp reload file')
            self.branches['temp'] = load_object('temp\\reload')
            o = self.branches['temp'].framework.output
            del self.branches['temp']
        print('cleaning up')
        while os.path.exists('temp\\reload.pkl'):
            try:
                if os.path.isfile('temp\\reload.pkl'):
                    os.remove('temp\\reload.pkl')
            except PermissionError:
                time.sleep(3)
        
        if branch.framework.production_only:
            branch.framework.output = o
            print('merging dictionaries')
            for i, chunk in enumerate(o):
                if i == 0:
                    continue
                for c in chunk.keys():
                    o[0][c] = np.concatenate([o[0][c], chunk[c]])
            branch.framework.output = pd.DataFrame(o[0])
            branch.framework.output.dropna(subset=['scenario'], inplace=True)
            # for c in ('gas_adj_unit', 'oil_adj_unit', 'ngl_adj_unit')
            print('saving output')
            save_output(branch.framework)
        stop = time.time()
        timer(start, stop)

    def actually_build_output(self):
        self.branches[list(self.branches.keys())[0]].framework.run_populate()

    def monte_carlo(self, branch_name):
        start = time.time()
        build_script_path = os.path.dirname(__file__)
        python_path = sys.executable
        print('preparing simulation run')
        p = subprocess.Popen(python_path + ' ' + build_script_path + '\\monte_carlo.py',
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        for line in p.stdout:
            print(line.rstrip())
        while not os.path.exists('temp\\reload.pkl'):
            time.sleep(5)
        if os.path.isfile('temp\\reload.pkl'):
            print('\nloading temp reload file')
            self.branches['temp'] = load_object('temp\\reload')
            e = self.branches['temp'].framework.econ_dists
            del self.branches['temp']
        print('cleaning up')
        while os.path.exists('temp\\reload.pkl'):
            try:
                if os.path.isfile('temp\\reload.pkl'):
                    os.remove('temp\\reload.pkl')
            except PermissionError:
                time.sleep(3)

        for i, sim in enumerate(e):
            if i == 0:
                continue
            for k in sim.keys():
                e[0][k] = np.concatenate([e[0][k], sim[k]])
        self.branches[branch_name].framework.econ_dists = pd.DataFrame(e[0])
        save_dists_to_excel(self.branches[branch_name], 'archive')
        save_dists_to_excel(self.branches[branch_name], 'main')
        save_dists_to_sql(self.branches[branch_name])
        stop = time.time()
        timer(start, stop)

    def actually_monte_carlo(self):
        self.branches[list(self.branches.keys())[0]].framework.run_mc()

    def autofit(self, branch_name):
        start = time.time()
        build_script_path = os.path.dirname(__file__)
        python_path = sys.executable
        print('preparing autoforecaster')
        p = subprocess.Popen(python_path + ' ' + build_script_path + '\\autofit.py',
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
        for line in p.stdout:
            print(line.rstrip())
        while not os.path.exists('temp\\reload.pkl'):
            time.sleep(5)
        if os.path.isfile('temp\\reload.pkl'):
            print('\nloading temp reload file')
            self.branches['temp'] = load_object('temp\\reload')
            w = self.branches['temp'].forecaster.well_dict
            l = self.branches['temp'].forecaster.log
            del self.branches['temp']
        print('cleaning up')
        while os.path.exists('temp\\reload.pkl'):
            try:
                if os.path.isfile('temp\\reload.pkl'):
                    os.remove('temp\\reload.pkl')
            except PermissionError:
                time.sleep(3)

        if isinstance(w, list):
            first = True
            for i, well in enumerate(w):
                if well:
                    if first:
                        w0 = well
                        first = False
                    else:
                        for k, v in well.items():
                            w0[k] = v
            self.branches[branch_name].forecaster.well_dict = w0
            del w0

        self.branches[branch_name].forecaster.prod_info = self.branches[branch_name].forecaster.build_prod_info()

        if isinstance(l, list):
            first = True
            for i, log in enumerate(l):
                if log['idp']:
                    if first:
                        l0 = log
                        first = False
                    else:
                        for k, v in log.items():
                            l0[k].extend(v)
            if not first:
                self.branches[branch_name].forecaster.log = l0
                del l0

        del w, l

    def actually_autofit(self):
        self.branches[list(self.branches.keys())[0]].forecaster.autofit()

## Move to forecasting module ##
    def update_forecasts(self, branch):
        print('updating production in forecasts')
        if branch.framework.economics is None:
            branch.framework.load_well_data()
        for w in branch.framework.well_dict.values():
            if w.forecast != w.idp:
                continue
            production = load_production(self.connection_dict, w)
            production.iloc[:-3]
            time_on = np.arange(1, production.shape[0] + 1)
            production['time_on'] = time_on
            production = production[['idp', 'time_on', 'prod_date',
                                     'gross_gas', 'gross_oil', 'gross_water']]
            production.rename(columns={'idp': 'forecast'}, inplace=True)
            update_forecast(self.connection_dict, w, production)


class Branch():
    def __init__(self, branch_name, tree, scenario, max_date):
        print('creating branch:\t\t\t'+branch_name)
        self.name = branch_name
        self.tree = tree
        if scenario is not None:
            self.scenario = self.branch_scenario(scenario)
        else:
            self.scenario = self.branch_scenario(self.name)
        self.properties = None
        self.schedule = None
        self.capacity = None
        self.framework = None
        self.forecaster = None
        if max_date:
            self.max_date = pd.Timestamp(max_date)
        else:
            self.max_date = None
        self.model = _dotdict()

    def __repr__(self):
        return self.name

    def branch_scenario(self, scenario):
        scenario = load_scenario(self, scenario)
        if not check_scenario(self, scenario):
            print('scenario check failed')
            return
        print(scenario)
        return scenario

    def forecast(self):
        self.forecaster = Forecaster(self)

    def build_schedule(self, schedule_file_path, gantt_start_date=date(2019, 1, 1),
                       gantt_years=3, show_gantt=True, verbose=False):
        print('\nbuilding schedule')
        self.schedule = Schedule(self,
                                 schedule_file_path,
                                 gantt_start_date,
                                 gantt_years,
                                 show_gantt)

    def load_schedule(self):
        print('\nloading schedule')
        self.schedule = load_object(self.tree.name + '\\' + self.scenario.schedule)
        if self.properties is None:
            self.properties = self.schedule.properties
        else:
            self.properties = pd.concat([self.properties, self.schedule.properties])
        for _, row in self.properties.iterrows():
            if row['propnum'] in self.model.keys():
                self.model[row['propnum']].schedule = self.schedule.well_dict[row['propnum']]
            else:
                self.model[row['propnum']] = _dotdict({
                                                       'tree': self.tree,
                                                       'branch': self,
                                                       'idp': row['propnum'],
                                                       'budget_type': row['budget_type'],
                                                       'well_name': row['bolo_well_name'],
                                                       'pad': row['pad'],
                                                       'short_pad': row['short_pad'],
                                                       'area': row['prospect'],
                                                       'project': None,
                                                       'project_id': None,
                                                       'properties': self.scenario.properties,
                                                       'schedule': self.schedule.well_dict[row['propnum']],
                                                       'schedule_inputs': self.scenario.schedule_inputs
                                                       })

    def load_framework(self):
        self.framework = Framework(self)

    def capacity_model(self):
        if self.capacity is None:
            self.capacity = Capacity(self)
            self.capacity.build_dataframe()
            self.capacity.story_gulch()
            self.capacity.offload()
            self.capacity.middle_fork()
        self.save_excel_model()
        print('generating pdf report')
        self.capacity.pdf_report(file_path='archive')
        self.capacity.pdf_report(file_path='main')

    def monte_carlo(self, num_simulations=1, uncertainty=None, risk=None):
        self.framework.uncertainty = uncertainty
        self.framework.risk = risk
        self.framework.num_simulations = num_simulations
        print('\nsaving temp load file')
        save_object(self, 'temp\\load')
        time.sleep(3)
        self.tree.monte_carlo(self.name)

    def autofit(self, overwrite=False, forecast_type=None):
        if self.forecaster is None:
            self.forecaster = Forecaster(self, overwrite, forecast_type)
        print('\nstarting autoforecaster')
        print('overwrite:\t\t\t\t' + str(overwrite))
        if forecast_type:
            print('forecast type:\t\t\t\t' + str(forecast_type))
        print('deleting old production forecast info')
        delete_prod_info(self.forecaster, overwrite, forecast_type)
        print('deleting old production forecasts')
        delete_prod_forecasts(self.forecaster, overwrite, forecast_type)
        print('\nsaving temp load file')
        save_object(self, 'temp\\load')
        time.sleep(3)
        self.tree.autofit(self.name)
        print('\nsaving new production forecast info')
        save_prod_info(self.forecaster, overwrite)
        print('saving log')
        save_log(self.forecaster)

    def dash_test(self):
        save_object(self, 'temp\\load')
        time.sleep(0.5)
        self.tree.dash_test(self.name)
        # run_server(self)

    def multigraph(self, properties=None):
        plot(self, properties)

    def save_excel_model(self):
        print('saving capacity model to excel')
        save_to_excel(self, file_path='archive')
        save_to_excel(self, file_path='main')

    def add_properties(self, idp=None, pad=None, short_pad=None,
                       area=None, scenario=None, project=None):
        if idp is not None:
            print('\nadding properties by id')
            properties = load_properties(self, idp=idp)
        elif pad is not None:
            if type(pad) == list:
                print('\nadding properties by pad:\t\t' + ', '.join(p for p in pad))
            else:
                print('\nadding properties by pad:\t\t' + pad)
            properties = load_properties(self, pad=pad)
        elif short_pad is not None:
            if type(short_pad) == list:
                print('\nadding properties by short_pad:\t\t' + ', '.join(p for p in short_pad))
            else:
                print('\nadding properties by short_pad:\t\t' + short_pad)
            properties = load_properties(self, short_pad=short_pad)
        elif area is not None:
            print('\nadding properties by area:\t\t' + ', '.join(a for a in area))
            properties = load_properties(self, area=area)
        elif self.scenario.area is not None:
            print('\nadding properties by area:\t\t' + ', '.join(a for a in area))
            properties = load_properties(self, area=area)            
        elif scenario is not None:
            print('\nadding properties by scenario')
            properties = load_properties(self, scenario=scenario)
        elif project is not None:
            print('\nadding properties by project')
            properties = load_properties(self, project=(project['name'], project['id']))
        elif self.scenario.project is not None:
            print('\nadding properties by project')
            project = self.scenario.project
            project_id = self.scenario.project_id
            properties = load_properties(self, project=(project, project_id))       

        if self.properties is None:
            self.properties = properties
        else:
            for p in self.properties.propnum:
                if p in properties.propnum.values:
                    tmp = properties[properties.propnum == p]
                    properties = properties[properties.propnum != p]
                    print(p, tmp.bolo_well_name.values[0],
                          tmp.short_pad.values[0], tmp.pad.values[0],
                          'already exists, skipping')
            self.properties = pd.concat([self.properties, properties])

        for p in self.properties.propnum.values:
            if p not in self.model.keys():
                tmp = properties[properties.propnum == p]
                tmp_sched = Well_Sched('base', p, tmp.bolo_well_name.values[0],
                                       tmp.short_pad.values[0], tmp['pad'].values[0],
                                       tmp.prospect, tmp.depth.values[0])
                tmp_sched.drill_start_date = tmp.drill_start_date
                tmp_sched.drill_end_date = tmp.drill_end_date
                tmp_sched.compl_start_date = tmp.compl_start_date
                tmp_sched.compl_end_date = tmp.compl_end_date
                tmp_sched.prod_start_date = tmp.first_prod_date
                self.model[p] = _dotdict({
                                         'tree': self.tree,
                                         'branch': self,
                                         'idp': p,
                                         'budget_type': tmp.budget_type.values[0],
                                         'well_name': tmp.bolo_well_name.values[0],
                                         'pad': tmp['pad'].values[0],
                                         'short_pad': tmp['short_pad'].values[0],
                                         'area': tmp['prospect'].values[0],
                                         'project': None,
                                         'project_id': None,
                                         'properties': self.scenario.properties,
                                         'schedule': tmp_sched,
                                         'schedule_inputs': None
                                          })

        print(len(properties.propnum.unique()), 'new properties added')

        if self.framework is not None:
            self.framework.load_well_data(properties)


    def load_output(self, scenario_name=None):
        start = time.time()
        print('loading output from sql')
        if scenario_name is None:
            scenario_name = self.name
        self.framework.output = load_output_from_sql(self, scenario_name)
        stop = time.time()
        timer(start, stop)

    def build_output(self, uncertainty=None, risk=None, delete_all=True):
        self.framework.uncertainty = uncertainty
        self.framework.risk = risk
        self.framework.delete_all = delete_all
        print('\nsaving temp load file')
        save_object(self, 'temp\\load')
        time.sleep(3)
        self.tree.build_output(self.name)

    def save_branch(self, name=None):
        if name is None:
            print('saving branch pickle')
            name = self.scenario.scenario
        save_object(self, name)
