from version1 import main as main1
from version2 import main as main2
from version3 import main as main3
from version4 import main as main4
from version5 import main as main5
from version6 import main as main6
from version7 import main as main7
from version8 import main as main8
from version9 import main as main9
from version11 import main as main11


if __name__ == '__main__':
    import sys

    version = 9
    if len(sys.argv) > 1:
        version = int(sys.argv[1])

    if version == 1:
        main1()
    if version == 2:
        main2()
    if version == 3:
        main3()
    if version == 4:
        main4()
    if version == 5:
        main5()
    if version == 6:
        main6()
    if version == 7:
        main7()
    if version == 8:
        main8()
    if version == 9:
        main9()
    if version == 11:
        main11()
