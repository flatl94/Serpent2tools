import os
import numpy as np
import math
import serpentTools
import sys
import copy
import itertools
import psutil
from mpi4py import MPI

class DetectorReader:
    """
    This class will transform the detector file into a dictionary whose keys
    are the detector names. For each detector, nested detectors appear, which
    keys identifies all the bins and error for a universe (u), cell (c), 
    material (m), lattice (l) and reaction channel (r) spacial and energy grids
    detector lattice type (cartesian, hexagonal lattices or cylindrical/spherical).
    The general structure of the ouput is recalled here:
    
    Output: dict;
        Output.keys(): name of all the detector:
        Output.values(): dict;
            Output[name_detector]:
            {'tallies': dict
                Output['tallies'].keys(): 'uA_cB_mC_lD_rE'
                Output['tallies'].values(): dict;
                    Bin and error value associated to the universe A, cell B, material C,
                    lattice D and reaction channel E. A,B,C,D,E are integers starting from
                    0 up to N-1 where N is the dimension reported in the dictionary
                    'dim' to the corresponding key.
                    keys():'base'
                    values():dict
                        Output['tallies']['base'].keys(): 'values', 'errors'
                        Output['tallies']['base'].values(): list
                            Output['tallies']['base'][j]: np.array((dim_x1,dim_x2,dim_x3))
                            where dim_x1, dim_x2, dim_x3 are the number of elements in the three dimension, and where j in [0, N_ene-1]
                            being N_ene the number of energy groups.
                        
            'grids': dict;
                keys(): str; name of the grid
                values(): np.array
                    np.array storing the the grid unmodified
            'type': str;
                'Cartesian','Hexagonal1','Hexagonal2','Spherical','Cylindrical'
            'dim': dict;
                keys():['ene','uni','cell','mat','lat','x3','x2','x1']
                values():int
                    The values express the number energy, universes,
                    cells, materials, lattices, x3, x2, x1 possible indices.
            }

    Args:
    
    _path_file : str
        - name of SERPENT output file containg the detectors to be analyzed
    
    _list_detectors : list
        - Is the name of the detector to analyze    
        
    """
    def __init__(self, _path_file, _list_detectors):
        
        # --- # FILE OPENING # --- #
        
        self.__file_reader = serpentTools.read(_path_file, reader='det')
        self.__output = {}
        
        array_cartesian = ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction', 'zmesh',
            'ymesh', 'xmesh']
        array_hexagonal = ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction', 'zmesh',
            'ycoord', 'xcoord']
        array_cylindrical = ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction', 'zmesh',
            'phi', 'rmesh']
        array_spherical = ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction', 'theta',
            'phi', 'rmesh']
        
        # --- # POPULATING THE OUTPUT FILE WITH THE DETECTOR DATA # --- #
        
        for det in _list_detectors:
            # --- # CREATING THE DETECTOR KEY # --- #
            
            self.__output[det] = {'tallies':{}}
            det_instance = self.__file_reader.detectors[det]
            
            # --- # CHECKING THE DETECTOR TYPE # --- #
            
            full_columns_name = np.array(det_instance.DET_COLS)
            list_columns_name = list(np.delete(full_columns_name,[0, 1, 11, 12]))
            if np.array_equal(list_columns_name,array_cartesian):
                detector_type = 'Cartesian'    
            if np.array_equal(list_columns_name,array_hexagonal):    
                detector_type = 'Hexagonal'    
            if np.array_equal(list_columns_name,array_cylindrical):    
                detector_type = 'Cylindrical'    
            if np.array_equal(list_columns_name,array_spherical):    
                detector_type = 'Spherical'
            
            self.__output[det]['type'] = detector_type
            
            # --- # CHECKING DETECTOR DIMENSION # --- #
            
            self.__output[det]['dim'] = {'energy':1,
                                         'universe':1,
                                         'cell':1,
                                         'material':1,
                                         'lattice':1,
                                         'reaction':1,
                                         'x3':1,
                                         'x2':1,
                                         'x1':1}
            
            dim_tester_keys = np.array(det_instance.indexes)
            dim_tester_values = np.array(det_instance.tallies.shape)
            dim_dict = {}
            for j in range(len(dim_tester_keys)):
                dim_dict[dim_tester_keys[j]] = dim_tester_values[j]
            
            for key, res in dim_dict.items():
                if key in ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction']:
                    self.__output[det]['dim'][key] = res
                elif key in ['zmesh','theta']:
                    self.__output[det]['dim']['x3'] = res
                elif key in ['ymesh','ycoord','phi']:
                    self.__output[det]['dim']['x2'] = res
                elif key in ['xmesh','xcoord','rmesh']:
                    self.__output[det]['dim']['x1'] = res
            
            # --- # OBTAINGING THE GRIDS OF THE FOR THE DETECTOR # --- #
            
            self.__output[det]['grids'] = {}
            for key_grid, res_grid in det_instance.grids.items():
                self.__output[det]['grids']['grid_'+key_grid] = res_grid
                
            # --- # POPULATING THE DICTIONARY WITH BINS AND ERROR OF THE DETECTOR # --- #
            
            dict_slice = {'energy':0,
                          'universe':0,
                          'cell':0,
                          'material':0,
                          'lattice':0,
                          'reaction':0}
            for dim_test, dim_value in self.__output[det]['dim'].items():
                if dim_value == 1 and dim_test in ['energy', 'universe', 'cell', 'material', 'lattice', 'reaction']:
                    dict_slice.pop(dim_test)
                    

            for uni in range(self.__output[det]['dim']['universe']):
                key_uni = 'u'+str(uni)
                
                if 'universe' in dict_slice.keys():
                    dict_slice['universe'] = uni
                for cell in range(self.__output[det]['dim']['cell']):
                    key_cell = 'c'+str(cell)
                    if 'cell' in dict_slice.keys():
                        dict_slice['cell'] = cell
                    for mat in range(self.__output[det]['dim']['material']):
                        key_mat = 'm'+str(mat)
                        if 'material' in dict_slice.keys():
                            dict_slice['material'] = mat
                        for lat in range(self.__output[det]['dim']['lattice']):
                            key_lat = 'l'+str(lat)
                            if 'lattice' in dict_slice.keys():
                                dict_slice['lattice'] = lat
                            for rch in range(self.__output[det]['dim']['reaction']):
                                key_rch = 'r'+str(rch)
                                if 'reaction' in dict_slice.keys():
                                    dict_slice['reaction'] = rch
                                
                                tally_key_dict = '_'.join([key_uni,key_cell,key_mat,key_lat,key_rch])
                                self.__output[det]['tallies'][tally_key_dict] = {'base':{'values':[],
                                                                                         'errors':[],}}
                                for ene in range(self.__output[det]['dim']['energy']):
                                    if 'energy' in dict_slice.keys():
                                        dict_slice['energy'] = ene
                                    
                                    values_mat = det_instance.slice(dict_slice).reshape((self.__output[det]['dim']['x3'],self.__output[det]['dim']['x2'],self.__output[det]['dim']['x1']))
                                    errors_mat = np.multiply(det_instance.slice(dict_slice),det_instance.slice(dict_slice,'errors')).reshape((self.__output[det]['dim']['x3'],self.__output[det]['dim']['x2'],self.__output[det]['dim']['x1']))
                                    
                                    self.__output[det]['tallies'][tally_key_dict]['base']['values'].append(np.transpose(values_mat,(2,1,0)))
                                    self.__output[det]['tallies'][tally_key_dict]['base']['errors'].append(np.transpose(errors_mat,(2,1,0)))
                                            

            
    def tallies_merge(self, _name_detector, _list_merge, _tally_keys={'universe':'all',
                                                                 'cell':'all',
                                                                 'material':'all',
                                                                 'lattice':'all',
                                                                 'reaction':'all'}, _composition = []):
        """
        This function creates additional keys in the 'tallies' dictionary storing data which have been merged with respect to the requested list of data.
        This function only perform universe-cell-material-reaction. The merge makes disappear the identification in the card. Example:
        
        Applying cell merging: 'u1_c[:]_m2_l0_r0' ---> 'u1_m2_l0_r0'
        Apllying cell and universe merging: 'u[:]_c[:]_m2_l0_r0' ---> 'm2_l0_r0'
        
        Args:
        _name_detector: str
            Name of the detector to which the merging procedure must be applied.
        _list_merge: list; 
            _list_merge[i]: str
            List of type of merging. Accepted keys are 'material', 'universe', 'cell' and/or 'reaction'. The order at which they are performed is not important.
        _tally_keys: dict;
            Dictionary which allows to speed up the operation by setting which combination of universe-cell-material-lattice-reaction is subjected to the operation.
            The value associated to each key must be either 'all' or an integer.
        _composition: list;
            Isotopic composition of the materials over which the reaction channels are reported. The order at which the isotopic composition of each element is reported,
            must be consistent with the methodology with which the detector was created.
            Optional in all other cases.
        """
        print('Performing uni/cell/mat/rch merge on detector: '+_name_detector+'...   ... merge type: '+str(_list_merge))
        
        # --- # CHECKING CONSISTENCY OF INPUT DATA # --- #
        if _name_detector not in self.__output.keys():
            raise NameError('Error! Detector not present in the database.')
        if 'reaction' in _list_merge:
            if self.__output[_name_detector]['dim']['reaction'] != len(_composition):
                raise ValueError('Error! The composition list length is different from the number of reaction channels.')
        for merge in _list_merge:
            if merge not in ['universe','material','cell','reaction']:
                raise ValueError('Error! Unable to perform the requested type of merging.')
            if _tally_keys[merge] != 'all':
                print('The selected merging technique is not consistent with the combination selected. All combinations will be investigated, instead.')
                _tally_keys[merge] = 'all'
        for key, res in _tally_keys.items():
            if res != 'all':
                if res > self.__output[_name_detector]['dim'][key]-1:
                    raise ValueError('Error! The requested '+
                                     key + ' is not present within the detector. Maximum accepted value is: '+(self.__output[_name_detector]['dim'][key]-1))
        # --- # MODULE TO CREATES KEYS TO BE PUT INSIDE THE DETECTOR # --- #
        dict_comb = {'universe':[],'cell':[],'material':[],'lattice':[],'reaction':[]}
        
        surv_list = []
        for i in ['universe','cell','material','lattice','reaction']:
            if i in _list_merge:
               dict_comb.pop(i, None)
            else:
                surv_list.append(i)
                if _tally_keys[i] != 'all':
                    dict_comb[i].append(i[0]+str(_tally_keys[i]))
                else:
                    for j in range(self.__output[_name_detector]['dim'][i]):
                        dict_comb[i].append(i[0]+str(j))
                        
        iterable_list = []                
        for key in surv_list:
            iterable_list.append(dict_comb[key])
        
        comb_list = list(itertools.product(*iterable_list))
        
        check_list = []
        for comb in comb_list:
            check_list.append(list(comb))
        
        # --- # MODULE FOR MERGING OF DATA # --- #
        dict_res = {}
        for key in range(len(check_list)):
            ckey = '_'.join(check_list[key])
            
            if ckey not in dict_res.keys():
                dict_res[ckey] = {'base':{'values':[np.zeros((self.__output[_name_detector]['dim']['x1'],
                                                              self.__output[_name_detector]['dim']['x2'],
                                                              self.__output[_name_detector]['dim']['x3']
                                                              ))]*self.__output[_name_detector]['dim']['energy'],
                                          'errors':[np.zeros((self.__output[_name_detector]['dim']['x1'],
                                                              self.__output[_name_detector]['dim']['x2'],
                                                              self.__output[_name_detector]['dim']['x3']
                                                              ))]*self.__output[_name_detector]['dim']['energy']}}

            for key_d, res_d in self.__output[_name_detector]['tallies'].items():
                
                if all(j in key_d for j in check_list[key]) and ckey != key_d:
                    for ene in range(self.__output[_name_detector]['dim']['energy']):
                        if 'reaction' not in _list_merge:
                            dict_res[ckey]['base']['values'][ene] = dict_res[ckey]['base']['values'][ene] + res_d['base']['values'][ene]
                            dict_res[ckey]['base']['errors'][ene] = dict_res[ckey]['base']['errors'][ene] + res_d['base']['errors'][ene]
                        else:
                            for rch in range(self.__output[_name_detector]['dim']['reaction']):
                                if 'r'+str(rch) in key_d:
                                    dict_res[ckey]['base']['values'][ene] = dict_res[ckey]['base']['values'][ene] + res_d['base']['values'][ene]*_composition[rch]
                                    dict_res[ckey]['base']['errors'][ene] = dict_res[ckey]['base']['errors'][ene] + res_d['base']['errors'][ene]*_composition[rch]
                        
        print('OLD KEYS: '+str(self.__output[_name_detector]['tallies'].keys()))                           
        for key_m, res_m in dict_res.items():
            print('...adding: '+key_m)
            self.__output[_name_detector]['tallies'][key_m] = copy.deepcopy(res_m)
        print('NEW KEYS: '+str(self.__output[_name_detector]['tallies'].keys()))        
            
    def bins_merge(self, _name_detector, _list_merge, _list_tally_keys='all', _list_mask_keys='all', _unc_prop = False):
        
        """
        This function creates additional keys for the selected combinations of universe-cell-material-lattice-reaction
        and performs bin merging across one or multiple dimension. The results have the same structure of the base detector,
        but the dimension of the arrays reflects the chosen merging technique.
        
        The new keys of the dictionary will have the following structure:
        'base' ---> 'base_' + key_merge, where key_merge is a string given by ''.join(list_merge). 
        'mask_'+name_mask ---> 'mask_' + name_mask + '_' + key_merge.
        
        Examples:
        'base_x1x2x3', 'base_x1x3', 'base_x3'
        
        Args:
        _name_detector: str
            Name of the detector to which the merging procedure must be applied.
        _list_merge: list; 
            _list_merge[i]: str
            List of type of merging. Accepted keys are 'x1', 'x2' and/or 'x3'. The order at which they are performed is not important.
        _list_tally_keys: list or 'all';
            List of dictionary keys of the self.__output[_name_detector]['tallies'] over which the operation must be performed.
            If instead of a list, the string 'all' is provided as input, the operation is performed on all the the dictionaries.
            Default value is 'all'.
        _list_mask_keys: list or 'all';
            List of dictionary bin keys self.__output[_name_detector]['tallies'][_list_dict_keys[i]] or if is set as 'all', all the bins are analyzed.
            Default value is 'all'.

        """
          
        print('Performing bin merge on detector: '+_name_detector+'...   ... merge type: '+str(_list_merge))
        # --- # CHECKING CONSISTENCY OF INPUT DATA # --- #
        
        if _name_detector not in self.__output.keys():
            raise NameError('Error! Detector not present in the database.')
        
        for i1 in _list_merge:
            if i1 not in ['x1','x2','x3']:
                raise ValueError('Error! Unable to perform the requested type of merging.')
        if _list_tally_keys != 'all' and isinstance(_list_tally_keys,list):
            for key_t in _list_tally_keys():
                if key_t not in self.__output[_name_detector]['tallies'].keys():
                    raise ValueError('Error! The requested combination of universe-cell-material-lattice and reaction, '+key_t+', does not exist.')
                if _list_mask_keys != 'all' and type(_list_mask_keys) is list:
                    for key_m in _list_mask_keys():
                        if key_m not in self.__output[_name_detector]['tallies'][key_m].keys():
                            raise ValueError('Error! The requested mask, '+key_m+', does not exist in: ' +key_t)
                else:
                    _list_mask_keys = list(self.__output[_name_detector]['tallies'][key_m].keys())      
        else:
            _list_tally_keys = list(self.__output[_name_detector]['tallies'].keys())
            for key_t in _list_tally_keys:
                if _list_mask_keys != 'all' and isinstance(_list_mask_keys,list):
                        for key_m in _list_mask_keys:
                            if key_m not in self.__output[_name_detector]['tallies'][key_t].keys():
                                raise ValueError('Error! The requested mask, '+key_m+', does not exist in: ' +key_t)
                else:
                    _list_mask_keys = list(self.__output[_name_detector]['tallies'][key_t].keys())
                    
        final_list_mask = []
        for key_m in _list_mask_keys:
            if any([x in key_m for x in ['x1','x2','x3']]):
                continue
            else:
                final_list_mask.append(key_m)
                    
        # --- # PERFORMING THE OPERATION OF BIN MERGING # --- #
        result_dict = {}
        for key_t, res_t in self.__output[_name_detector]['tallies'].items():
            if key_t in _list_tally_keys:
                result_dict[key_t] = {}
                for key_m, res_m in res_t.items():
                    if key_m in final_list_mask:
                        new_mask_key = key_m + '_' + ''.join(_list_merge)
                        
                        result_dict[key_t][new_mask_key] = {'values':[],'errors':[]}
                        
                        sum_dim = []
                        if 'x1' in _list_merge:
                            dim_fin_x1 = 1
                            sum_dim.append(0)
                        else:
                            dim_fin_x1 = self.__output[_name_detector]['dim']['x1']
                            
                        if 'x2' in _list_merge:
                            dim_fin_x2 = 1
                            sum_dim.append(1)
                        else:
                            dim_fin_x2 = self.__output[_name_detector]['dim']['x2']
                            
                        if 'x3' in _list_merge:
                            dim_fin_x3 = 1
                            sum_dim.append(2)
                        else:
                            dim_fin_x3 = self.__output[_name_detector]['dim']['x3']
                            
                        sum_dim = tuple(sum_dim)
                        
                        for ene in range(self.__output[_name_detector]['dim']['energy']):
                            # --- # HANDLING OF VALUES AND ERRORS# --- #
                            res_values = np.sum(res_m['values'][ene], sum_dim)
                            
                            res_values_mat = np.reshape(res_values, (dim_fin_x1, dim_fin_x2, dim_fin_x3))
                            result_dict[key_t][new_mask_key]['values'].append(res_values_mat)
                        
                            res_errors = np.sum(res_m['values'][ene], sum_dim)
                            res_errors_mat = np.reshape(res_errors, (dim_fin_x1, dim_fin_x2, dim_fin_x3))                 
                            result_dict[key_t][new_mask_key]['errors'].append(res_errors_mat)
        for key_t, res_t in result_dict.items():
            for key_m, res_m in res_t.items():
                self.__output[_name_detector]['tallies'][key_t][key_m] = copy.deepcopy(res_m)
                
        print('operazione conclusa...')
                                        
    def set_geometrical_mask(self, _name_detector, mask, _list_tally_keys='all'):
        """
        This function will set a geometrical mask. Those values which do not follows the rules of the mask will be set to 0.0.
        These operation must be performed BIN MERGING for a correct evaluation of data.
        For each uni-cell-mat-lat-rch combination selected, this function will use the 'base' values and errors to create a ney dictionary key
        storing the data opportunately masked accordint to the selected mask.
        
        Args:
        _name_detector: str;
            Name of the detector
            
        mask: dict;
            mask.keys(): ['type','geom_params','name','where']
            mask['name']: str;
                Name to be assigned to the bask
            mask['type']: str;
                Type of mask. The currently accepted options are 'cyl' or 'prism'
                If mask['type'] == 'prism':
                    The mask created is a prism-like polyhedra.
                If mask['type'] == 'cyl':
                    The mask created is a cylinder
            mask['geom_params']: dict;
                Dictionary storing the geometrical information to construct the geometrical mask.
                It has different keys depending on the type of _mask.
                If mask['type'] == 'prism':
                    mask['geom_params'].keys(): ['verts','zmin','zmax']
                        mask['geom_params']['verts']: list;
                        List sotring coordinates of the vertices.
                            mask['geom_params']['verts'][i]: tuple;
                                mask['geom_params']['verts'][i][0]: float; x-coordinate of the vertex;
                                mask['geom_params']['verts'][i][1]: float; y-coordinate of the vertex;
                If mask['type'] == 'cyl':
                    mask['geom_params'].keys(): ['center','radius','zmin','zmax'];
                        mask['geom_params']['center']: tuple;
                            Coordinates of the center of the circle.
                            mask['geom_params']['center'][0]: float; x-coordinate of the center;
                            mask['geom_params']['center'][1]: float; y-coordinate of the center;
                mask['geom_params']['zmin']: float;
                    z-coordinate of the horizontal plane constituting the lower boundary surface.
                    If the key is not present in the dictionary, the coordinates will not be checked with respect to this plane.
                mask['geom_params']['zmax']: float;
                    z-coordinate of the horizontal plane constituting the jupper boundary surface.
                    If the key is not present in the dictionary, the coordinates will not be checked with respect to this plane.
            mask['where']: str;
                Selected if the data to be set to zero are within or outside the selected volume. 'inside' or 'outside' are the selected possibilities.
        _list_tally_keys: list or 'all';
            List of dictionary keys of the self.__output[_name_detector]['tallies'] over which the operation must be performed.
            If instead of a list, the string 'all' is provided as input, the operation is performed on all the the dictionaries.
            Default value is 'all'.
        """
        # --- # CHECK CONSISTENCY OF INPUT DATA # --- #
        from package_utilities.geometrical_tools import GeometryMask
        GM = GeometryMask(mask)
        
        if _name_detector not in self.__output.keys():
            raise NameError('Error! Detector not present in the database.')
        if _list_tally_keys != 'all' and isinstance(_list_tally_keys,list):
            for key_t in _list_tally_keys():
                if key_t not in self.__output[_name_detector]['tallies'].keys():
                    raise ValueError('Error! The requested combination of universe-cell-material-lattice and reaction, '+key_t+', does not exist.')
        elif _list_tally_keys == 'all':
            _list_tally_keys = list(self.__output[_name_detector]['tallies'].keys())
        else:
            raise ValueError('Error! "all" or list of tally key are the only accepted input data.')
        # --- # APPLYING THE GEOOMETRY MASK # --- #
        if 'ucs' not in self.__output[_name_detector].keys():
            self.set_ucs(_name_detector)
        for key_t, res_t in self.__output[_name_detector]['tallies'].items():
            if key_t in _list_tally_keys:
                res_t[mask['name']] = {'values':[],'errors':[]}
                for ene in range(self.__output[_name_detector]['dim']['energy']):
                    res_t[mask['name']]['values'].append(GM.set_mask(res_t['base']['values'][ene],self.__output[_name_detector]['ucs'],mask['where']))
                    res_t[mask['name']]['errors'].append(GM.set_mask(res_t['base']['errors'][ene],self.__output[_name_detector]['ucs'],mask['where']))
                
    def set_ucs(self, _name_detector):
        """
        This class will use the spational grids to create a dictionary key called 'ucs' (universal coordinate system)
        in which the coordinates of each bin is reported in the format
        
        self.__output[_name_detector]['ucs']: np.array((n_x1, n_x2, n_x3))
            self.__output[_name_detector]['ucs'][j1,j2,j3]: tuple
                self.__output[_name_detector]['ucs'][j1,j2,j3][0]: float; x-coordinate
                self.__output[_name_detector]['ucs'][j1,j2,j3][1]: float; y-coordinate
                self.__output[_name_detector]['ucs'][j1,j2,j3][2]: float; z-coordinate

        Args:
            _name_detector (_type_): _description_
        """
        # --- # CHECKING CONSISTENCY OF INPUT DATA # --- #
        print('Setting coordinates for detector: '+_name_detector)
        if _name_detector not in self.__output.keys():
            raise NameError('Error! Detector not present in the database.')
        
        array_coords = np.zeros((self.__output[_name_detector]['dim']['x1'],
                                 self.__output[_name_detector]['dim']['x2'],
                                 self.__output[_name_detector]['dim']['x3']),dtype=object)
        count_hexa = 0
        for j1 in range(self.__output[_name_detector]['dim']['x1']):
            for j2 in range(self.__output[_name_detector]['dim']['x2']):
                for j3 in range(self.__output[_name_detector]['dim']['x3']):
                    array_coords[j1,j2,j3] = [0.0, 0.0, 0.0]
                    # --- # ASSIGNING CARTESIAN COORDINATES # --- #
                    if self.__output[_name_detector]['type'] == 'Cartesian':
                        if 'grid_X' in self.__output[_name_detector]['grids'].keys():
                            array_coords[j1,j2,j3][0] = self.__output[_name_detector]['grids']['grid_X'][j1,2]
                        if 'grid_Y' in self.__output[_name_detector]['grids'].keys():
                            array_coords[j1,j2,j3][1] = self.__output[_name_detector]['grids']['grid_Y'][j2,2]
                        if 'grid_Z' in self.__output[_name_detector]['grids'].keys():
                            array_coords[j1,j2,j3][2] = self.__output[_name_detector]['grids']['grid_Z'][j3,2]
                    # --- # ASSIGNING HEXAGONAL COORDINATES # --- #
                    elif self.__output[_name_detector]['type'] == 'Hexagonal':
                        if 'grid_COORD' in self.__output[_name_detector]['grids'].keys():
                            array_coords[j1,j2,j3][0] = self.__output[_name_detector]['grids']['grid_COORD'][count_hexa,0]
                            array_coords[j1,j2,j3][1] = self.__output[_name_detector]['grids']['grid_COORD'][count_hexa,1]
                        if 'grid_Z' in self.__output[_name_detector]['grids'].keys():
                            array_coords[j1,j2,j3][2] = self.__output[_name_detector]['grids']['grid_Z'][j3,2]
                    array_coords[j1,j2,j3] = tuple(array_coords[j1,j2,j3])
                count_hexa = count_hexa+1            
                
        # --- # CYLINDRICAL AND SPHERICAL COORDINATE SYSTEM ARE YET TO BE WRITTEN # --- #        
        self.__output[_name_detector]['ucs'] = array_coords
            
    def get_output(self):
        """
        This function returns the output.

        """
        results = self.__output
        return results
            

    
