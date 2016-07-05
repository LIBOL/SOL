#! /usr/bin/evn python
"""parameter space for cross validation"""
import re
import sys

class SearchItem(object):
    """the grid search item, each grid item is a paramter with its parameter space
    """
    __slots__ = ('name','start_val','step_val','end_val','size')

    def __init__(self, name, start_val, step_val, end_val):
        """Create a new search item
        Parameters:
        name: string
            name of the parameter
        start_val: string or float
            start search value
        step_val: string or float
            search step 
        end_val: string or float
            end search value 
        """
        self.name = name
        self.start_val = float(start_val)
        self.step_val = float(step_val)
        if self.step_val == 1:
            raise ValueError('step value should not be 1')
        self.end_val = float(end_val)

        #calculate the size of the search space
        self.size = 0
        if self.end_val > self.start_val:
            val = self.start_val
            while val <= self.end_val:
                val *= self.step_val
                self.size += 1

    def __getitem__(self, index):
        """get one search value
        Parameteres:
        index: int
            index of the value in all search space of this parameter
        """
        ret = self.start_val
        while index > 0:
            ret *= self.step_val
            index -= 1
        return (self.name, ret)

    def __str__(self):
        return 'param: {0} range: {1}:{2}:{3}'\
                .format(self.name,self.start_val, self.step_val,self.end_val)
    
class SearchSpace(object):
    """Search space of all parameters
    """
    #dim: number of parameters to search
    #size: number of grid items in the search space
    #search_space: search space
    __slots__ = ('dim','size','search_space')

    def __init__(self, param_str):
        """create a search space with the given parameter string
        Parameters:
        cv_params: list[(name, range)]
            parameter string, with format like '[('a', '1:2:8'),(b, '2:2:16')]'
        """
        self.__parse_cv_params(param_str)

        self.dim = len(self.search_space)

        if self.dim > 0:
            self.size = reduce(lambda x, y: x * y, [item.size for item in self.search_space])
        else:
            self.size = 0

    def get_param(self, grid_item_id):
        param  = []
        for j in range(0,self.dim):
            dim_size = self.search_space[j].size
            coor = grid_item_id % dim_size
            grid_item_id = int(grid_item_id / dim_size)

            param.append(self.search_space[j][coor])
        return param

    def __parse_cv_params(self,cv_params):
        """parse the search space from parameter string
        Parameters:
        cv_params: list[(name, range)]
            parameter string, with format like '[('a', '1:2:8'),(b, '2:2:16')]'
        """
        #detect param
        self.search_space = []
        for param_name, param_range in cv_params:
            search_res  = param_range.strip().split(':')
            if len(search_res) == 3:
                self.search_space.append(SearchItem(param_name,
                        search_res[0],
                        search_res[1],
                        search_res[2]))

            else:
                raise ValueError('incorrect input parameter {0}'.format(param_range))

if __name__ == '__main__':
    param_space = [('a', '1:2:16'), ('b', '0.5:2:10')]
    ss = SearchSpace(param_space)
    for k in range(0,ss.size):
        cmd = ss.get_param(k)
        print cmd
