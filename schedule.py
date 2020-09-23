from .utils import *
from .utils import _dotdict
import time
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import pickle
pd.options.display.float_format = '{:,.2f}'.format


class Schedule():
    def __init__(self, branch, schedule_file_path, gantt_start_date='1/1/2019',
                 gantt_years=3, show_gantt=True):
        print('initializing schedule')
        self.branch = branch
        if branch.scenario.schedule is None:
            print('no schedule scenario provided')
            return
        if branch.scenario.schedule_inputs is None:
            print('no schedule inputs scenario provided')
            return
        self.name = branch.scenario.schedule
        self.schedule_path = schedule_file_path
        self.schedule_df = pd.read_excel(self.schedule_path,
                                         dtype={'rig': str,
                                                'pad': str,
                                                'schedule_inputs': str,
                                                'pad_start_date': 'datetime64[ns]',
                                                'pad_end_date': 'datetime64[ns]',
                                                'conductor_days': float,
                                                'mob_in_days': float,
                                                'drill_days': float,
                                                'mob_out_days': float,
                                                'logging_days': float,
                                                'build_fac_days': float,
                                                'frac_days': float,
                                                'flowback_days': float,
                                                'pod_size': int,
                                                'drill_start_date': 'datetime64[ns]',
                                                'drill_end_date': 'datetime64[ns]',
                                                'compl_start_date': 'datetime64[ns]',
                                                'compl_end_date': 'datetime64[ns]',
                                                'prod_start_date': 'datetime64[ns]',
                                                }
                                         )
        self.schedule_dates = pd.DataFrame(columns=['scenario', 'idp', 'drill_start_date',
                                                    'drill_end_date', 'compl_start_date',
                                                    'compl_end_date', 'prod_start_date'])

        self.gantt_start_date = pd.Timestamp(gantt_start_date)
        self.gantt_end_date = gantt_start_date + relativedelta(years=+gantt_years)
        self.show_gantt = show_gantt
        
        self.build_dictionaries()
        self.load_schedule_inputs()
        self.calc_dates()
        self.gantt_chart()
        self.save_schedule()

    def __repr__(self):
        return self.name

    def build_dictionaries(self):
        print('building rig and pad dictionaries')
        self.rig_dict = {}
        self.pad_dict = {}
        self.well_dict = {}
        self.schedule_df.loc[:, 'rig'] = self.schedule_df.loc[:, 'rig'].str.upper()
        self.schedule_df.loc[:, 'pad'] = self.schedule_df.loc[:, 'pad'].str.upper()
        self.schedule_df.loc[:, 'input_group'] = self.schedule_df.loc[:, 'input_group'].str.upper()
        for _, row in self.schedule_df.iterrows():
            if row['rig']!= 'DUC':
                if row['rig'] not in self.rig_dict.keys():
                    if pd.isnull(row['drill_start_date']):
                        print('rig:', row['rig'], 'pad:', row['pad'],
                              'first drill start date must be defined')
                        return
                    self.rig_dict[row['rig']] = Rig(row['rig'],
                                                    row['drill_start_date'])
            if row['pad'] in self.pad_dict.keys():
                print('duplicate pad:', row['pad'])
                return
            else:
                if pd.isnull(row['pod_size']):
                    print('rig:', row['rig'], 'pad:', row['pad'], 'pod size is not populated')
                    return
                if row['pod_size'] < 1:
                    print('rig:', row['rig'], 'pad:', row['pad'], 'pod size is less than one:', row['pod_size'])
                    return
                self.pad_dict[row['pad']] = Pad(row['pad'],
                                                self.rig_dict[row['rig']],
                                                row['input_group'],
                                                row['pod_size'])
                self.rig_dict[row['rig']].add_pad(self.pad_dict[row['pad']])
        print('loading properties')
        if self.branch.scenario.properties is None:
            print('no properties scenario provided')
            return
        self.properties = load_schedule_properties(self)
        self.properties.sort_values(by=['pad', 'schedule_order'], inplace=True)
        if self.branch.properties is None:
            self.branch.properties = self.properties
        else:
            self.branch.properties = pd.concat([self.branch.properties, self.properties])
        print('building well dictionary')
        for _, row in self.properties.iterrows():
            self.well_dict[row['propnum']] = Well_Sched(schedule=self,
                                                        idp=row['propnum'],
                                                        well_name=row['bolo_well_name'],
                                                        short_pad=row['short_pad'],
                                                        pad=self.pad_dict[row['pad']],
                                                        area=row['prospect'],
                                                        depth=row['depth'])
            self.pad_dict[row['pad']].add_well(self.well_dict[row['propnum']])

            if row['propnum'] in self.branch.model.keys():
                self.branch.model[row['propnum']].schedule = self.well_dict[row['propnum']]
                self.branch.model[row['propnum']].well_name = row['bolo_well_name']
                self.branch.model[row['propnum']].pad = rpw['pad']
                self.branch.model[row['propnum']].short_pad = row['short_pad']
                self.branch.model[row['propnum']].area = row['prospect']
            else:
                self.branch.model[row['propnum']] = _dotdict({
                                                             'tree': self.branch.tree,
                                                             'branch': self.branch,
                                                             'idp': row['propnum'],
                                                             'well_name': row['bolo_well_name'],
                                                             'pad': row['pad'],
                                                             'short_pad': row['short_pad'],
                                                             'area': row['prospect'],
                                                             'project': None,
                                                             'project_id': None,
                                                             'properties': self.branch.scenario.properties,
                                                             'schedule': self.well_dict[row['propnum']],
                                                             'schedule_inputs': self.branch.scenario.schedule_inputs
                                                             })

        self.schedule_dates['scenario'] = pd.Series([self.name]*len(self.well_dict.keys()))
        self.schedule_dates['idp'] = pd.Series(list(self.well_dict.keys()))
             
    def load_schedule_inputs(self):
        print('loading schedule inputs')
        self.schedule_inputs = load_schedule_inputs(self.branch)
        for pad in self.pad_dict.keys():
            s = self.schedule_inputs
            inputs = s[s.input_group == self.pad_dict[pad].input_group]

            for _, row in inputs.iterrows():

                if row['parameter'] == 'pad_build_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].set_build_time(float(row['value']))
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_build_time_by_num_wells(float(row['value']))

                if row['parameter'] == 'conductor_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].rig.set_conductors(float(row['value']))

                if row['parameter'] == 'mob_in_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].rig.set_mob_in(float(row['value']))

                if row['parameter'] == 'drill_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].set_drill_time(float(row['value']))
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_drill_time_by_depth(float(row['value']))

                if row['parameter'] == 'mob_out_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].rig.set_mob_out(float(row['value']))

                if row['parameter'] == 'build_fac_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].set_fac_time(float(row['value']))
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_fac_time_per_well(float(row['value']))

                if row['parameter'] == 'logging_days':
                    if row['unit'] == 'fix':
                        self.pad_dict[pad].set_logging_time(float(row['value']))
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_logging_time_per_well(float(row['value']))

                if row['parameter'] == 'frac_days':
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_compl_time(float(row['value']))

                if row['parameter'] == 'flowback_days':
                    if row['unit'] == 'var':
                        self.pad_dict[pad].set_flowback_time(float(row['value']))
                
            s = self.schedule_df[self.schedule_df['pad'] == pad]
            if not pd.isnull(s.pad_start_date.values[0]):
                self.pad_dict[pad].build_start = pd.Timestamp(s.pad_start_dates[0])
            if not pd.isnull(s.pad_end_date.values[0]):
                self.pad_dict[pad].build_finish = pd.Timestamp(s.pad_end_date.values[0])
            if not pd.isnull(s.conductor_days.values[0]):
                self.pad_dict[pad].conductors = s.conductor_days.values[0]
            if not pd.isnull(s.mob_in_days.values[0]):
                self.pad_dict[pad].mob_in = s.mob_in_days.values[0]
            if not pd.isnull(s.drill_days.values[0]):
                self.pad_dict[pad].set_drill_time(s.drill_days.values[0])
            if not pd.isnull(s.mob_out_days.values[0]):
                self.pad_dict[pad].mob_out = s.mob_out_days.values[0]
            if not pd.isnull(s.logging_days.values[0]):
                self.pad_dict[pad].set_logging_time_per_well(s.logging_days.values[0])
            if not pd.isnull(s.build_fac_days.values[0]):
                self.pad_dict[pad].set_fac_time(s.build_fac_days.values[0])
            if not pd.isnull(s.frac_days.values[0]):
                self.pad_dict[pad].set_compl_time(s.frac_days.values[0])
            if not pd.isnull(s.flowback_days.values[0]):
                self.pad_dict[pad].set_flowback_time(s.flowback_days.values[0])
            if not pd.isnull(s.drill_start_date.values[0]):
                self.pad_dict[pad].drill_start = pd.Timestamp(s.drill_start_date.values[0])
            if not pd.isnull(s.drill_end_date.values[0]):
                self.pad_dict[pad].drill_finish = pd.Timestamp(s.drill_end_date.values[0])
            if not pd.isnull(s.compl_start_date.values[0]):
                self.pad_dict[pad].compl_start = pd.Timestamp(s.compl_start_date.values[0])
            if not pd.isnull(s.compl_end_date.values[0]):
                self.pad_dict[pad].compl_finish = pd.Timestamp(s.compl_end_date.values[0])
            if not pd.isnull(s.prod_start_date.values[0]):
                self.pad_dict[pad].prod_start = pd.Timestamp(s.prod_start_date.values[0])                 

    def calc_dates(self):
        print('calculating drill dates')
        calc_drill_dates(self)
        print('calculating completion dates')
        calc_compl_dates(self)
        print('calculating start dates')
        calc_start_dates(self)
        print('saving schedule')
        save_schedule(self)
        save_pad_schedule(self)

    def gantt_chart(self):
        print('creating gantt chart')
        create_gantt_df(self)
        create_gantt_chart(self, 'main')
        create_gantt_chart(self, 'archive')

    def save_schedule(self):
        print('saving schedule pickle')

        pkl_name = str(self.branch.tree.name + 
                       '\\archive\\' + self.name +
                       '_schedule_' + self.branch.tree.run_time)
        save_object(self, pkl_name)

        pkl_name = str(self.branch.tree.name + '\\' + self.name)
        save_object(self, pkl_name)

class Rig():
    def __init__(self, rig_name, start_date):
        self.rig_name = rig_name
        self.start = start_date
        self.num_pads = 0
        self.pad_list = []
        self.conductors = 14
        self.mob_in = 7
        self.mob_out = 7

    def __repr__(self):
        print_dict = self.__dict__.copy()
        del print_dict['pad_list']
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        for c in print_df.index:
            if 'start' in c:
                try:
                    print_df.loc[c] = print_df.loc[c].dt.strftime('%x')
                except:
                    continue
        return pretty_print(print_df)

    def add_pad(self, pad):
        self.pad_list.append(pad)
        pad.rig_name = self.rig_name
        self.num_pads += 1

    def drop_pad(self, pad):
        self.pad_list.remove(pad)
        pad.rig_name = None
        self.num_pads -= 1

    def pads(self):
        if not self.pad_list:
            print('No pads assigned to the', self.rig_name, 'rig.')
        else:
            for idx, pad in enumerate(self.pad_list):
                print(idx+1, '-', pad.pad_name)

    def pad_count(self):
        if self.num_pads == 0:
            print('No pads assigned to the', self.rig_name, 'rig.')
        elif self.num_pads == 1:
            print('1 pad assigned to the', self.rig_name, 'rig.')
        else:
            print(self.num_pads, 'pads assigned to the', self.rig_name, 'rig.')

    def set_conductors(self, set_conductors_days):
        self.conductors = float(round(set_conductors_days, 1))
        for pad in self.pad_list:
            pad.conductors = self.conductors

    def set_mob_in(self, mob_in_days):
        self.mob_in = float(round(mob_in_days, 1))
        for pad in self.pad_list:
            pad.conductors = self.conductors

    def set_mob_out(self, mob_out_days):
        self.mob_out = float(round(mob_out_days, 1))
        for pad in self.pad_list:
            pad.conductors = self.conductors

    def set_drill_time(self, drill_time=10.0):
        for pad in self.pad_list:
            for well in pad.well_list:
                well.set_drill_time(drill_time)

    def set_drill_time_by_depth(self, feet_per_day=1600):
        for pad in self.pad_list:
            for well in pad.well_list:
                well.set_drill_time_by_depth(feet_per_day)


class Pad():
    def __init__(self, pad_name, rig, input_group, pod_size):
        self.pad_name = pad_name
        self.rig = rig
        self.input_group = input_group
        self.num_wells = 0
        self.well_list = []
        self.build_pad = 30
        self.pod_size = pod_size
        self.conductors = self.rig.conductors
        self.mob_in = self.rig.mob_in
        self.mob_out = self.rig.mob_out
        self.log_pad = 0.5
        self.build_facilities = 60
        self.build_start = None
        self.build_finish = None
        self.drill_start = None
        self.drill_finish = None
        self.compl_start = None
        self.compl_finish = None
        self.prod_start = None
        self.prod_finish = None

    def __repr__(self):
        print_dict = self.__dict__.copy()
        del print_dict['well_list']
        print_df = pd.DataFrame(print_dict, index=[0]).transpose()
        for c in print_df.index:
            if 'start' in c or 'finish' in c:
                try:
                    print_df.loc[c] = print_df.loc[c].dt.strftime('%x')
                except:
                    continue
        return pretty_print(print_df)

    def add_well(self, well):
        self.well_list.append(well)
        self.num_wells += 1

    def drop_well(self, well):
        self.well_list.remove(Well_Sched)
        well.pad_name = None
        self.num_wells -= 1

    def wells(self):
        if not self.well_list:
            print('No wells assigned to the', self.pad_name, 'pad.')
        else:
            name_list = sorted([w.well_name for w in self.well_list])
            for idx, well in enumerate(name_list):
                print(idx+1, '-', well)

    def well_count(self):
        if self.num_wells == 0:
            print('No wells assigned to the', self.pad_name, 'pad.')
        elif self.num_wells == 1:
            print('1 well assigned to the', self.pad_name, 'pad.')
        else:
            print(self.num_wells, 'wells assigned to the',
                  self.pad_name, 'pad.')

    def set_build_time(self, construct_days=30.0):
        self.build_pad = float(round(construct_days, 1))

    def set_build_time_by_num_wells(self, construct_days_per_well=2.0):
        self.build_pad = float(round(construct_days_per_well *
                                     self.num_wells, 1))

    def set_fac_time(self, fac_days=60.0):
        self.build_facilities = round(float(fac_days), 1)

    def set_fac_time_by_num_wells(self, fac_days_per_well=2.0):
        self.build_facilities = float(round(fac_days_per_well *
                                            self.num_wells, 1))

    def set_drill_time(self, drill_time=10.0):
        for well in self.well_list:
            well.set_drill_time(drill_time)

    def set_drill_time_by_depth(self, feet_per_day=1600):
        for well in self.well_list:
            well.set_drill_time_by_depth(feet_per_day)

    def set_logging_time(self, logging_days):
        self.log_pad = float(round(logging_days, 1))

    def set_logging_time_per_well(self, logging_days_per_well):
        self.log_pad = float(round(logging_days_per_well*self.num_wells, 1))

    def set_compl_time(self, compl_days):
        for well in self.well_list:
            well.set_compl_time(compl_days)

    def set_flowback_time(self, flowback_days):
        for well in self.well_list:
            well.set_flowback_time(flowback_days)

    def set_pod_size(self, pod_size):
        self.pod_size = pod_size


class Well_Sched():
    def __init__(self, schedule, idp, well_name, short_pad, pad, area, depth):
        self.schedule = schedule
        self.idp = idp
        self.well_name = well_name
        self.short_pad = short_pad
        self.pad = pad
        self.area = area
        self.depth = depth
        self.drill_time = 10.0
        self.compl_time = 3.
        self.flowback_time = 30.
        self.buffer = 0.
        self.drill_start_date = None
        self.drill_end_date = None
        self.compl_start_date = None
        self.compl_end_date = None
        self.prod_start_date = None

    def __repr__(self):
        print_df = pd.DataFrame(self.__dict__, index=[0]).transpose()
        for c in print_df.index:
                try:
                    print_df.loc[c] = print_df.loc[c].dt.strftime('%x')
                except:
                    continue
        return pretty_print(print_df)

    def set_drill_time(self, drill_time=10.0):
        self.drill_time = round(drill_time, 1)

    def set_drill_time_by_depth(self, feet_per_day=1600):
        self.drill_time = round(self.depth / feet_per_day, 1)

    def set_buffer(self, num_days=0):
        self.buffer = round(num_days, 1)

    def set_compl_time(self, compl_days):
        self.compl_time = float(round(compl_days, 1))

    def set_flowback_time(self, flowback_days):
        self.flowback_time = float(round(flowback_days, 1))
