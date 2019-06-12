#! /usr/bin/env python

import time


class timer(object):
    """A timer class
    """

    def __init__(self):
        """Constructor
        """
        self.clocks = {}

    def start(self, name):
        if not name in list(self.clocks.keys()):
            self.clocks[name] = {}
            self.clocks[name]['total'] = 0
        self.clocks[name]['start'] = time.time()

    def stop(self, name):
        self.clocks[name]['total'] += time.time() - self.clocks[name]['start']

    def __str__(self):
        output = ''
        for name in sorted(self.clocks.keys()):
            output += '%s: %.2e\n' % (name, self.clocks[name]['total'])
        return output


def test_timer():
    import sys
    myTimer = timer()
    myTimer.start('main')
    time.sleep(2)
    myTimer.stop('main')
    print(myTimer)


if __name__ == '__main__':
    test_timer()
