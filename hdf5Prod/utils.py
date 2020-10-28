# -*- coding: utf-8 -*-

"""
Common utility methods for egamma project.
"""

import os
import time
import logging as log
from subprocess import call



def mkdir (path):
    """
    Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        # print( "mdkir: Creating output directory:\n  {}".format(path) )
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass

    return



def cautious_remove (path):
    """
    ...
    """
    if path.startswith('/') or '*' in path:
        log.info("cautious_remove: Refusing to remove {}".format(path))
    else:
        log.debug("cautious_remove: Removing.")
        call(['rm', path])
        pass
    pass



def unique_tmp (path):
    """
    Utility script to create a unique, temporary file path.
    """
    ID = int(time.time() * 1E+06)
    basedir = '/'.join(path.split('/')[:-1])
    filename = path.split('/')[-1]
    filename = 'tmp.{:s}.{:d}'.format(filename, ID)
    return '{}/{}'.format(basedir, filename)
