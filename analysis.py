from make_parameters import Paramgroup as PG
import numpy as np
def externalinfluence():
    pg = PG()
    pg.default("general:runtime", 20)
    pg.default("species2:adsorbable", [True, False])
    pg.default('general:init_f', np.linspace(1,95,7)/100)
    pg.default('substrate:max', 6)
    pg.default('species2:mu', 24.5*0.9)

    pg.add('basic', 'substrate:max', [2,4,6,8])
    pg.add('cost', 'species2:mu', 24.5*np.linspace(80,100,4)/100)

    pg.set_repeats(10)
    return pg

if __name__ == '__main__':
    print(len(externalinfluence()))
    externalinfluence().print()