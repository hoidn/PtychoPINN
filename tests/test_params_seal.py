import logging
from ptycho import params


def test_post_seal_nonwhitelist_write_warns(caplog):
    prev = params.get('gridsize', 1)
    try:
        params.unseal()
        params.set('gridsize', 1)   # pre-seal write: must NOT warn
        params.seal()
        with caplog.at_level(logging.WARNING, logger='ptycho.params'):
            params.set('gridsize', 2)            # post-seal, NOT whitelisted -> WARN
        assert any('post-seal' in r.message and r.levelno == logging.WARNING
                   for r in caplog.records), "expected a post-seal warning for 'gridsize'"
    finally:
        params.unseal()
        params.set('gridsize', prev)


def test_post_seal_whitelisted_write_silent(caplog):
    had_is = 'intensity_scale' in params.cfg
    prev_is = params.cfg.get('intensity_scale')
    try:
        params.unseal()
        params.seal()
        with caplog.at_level(logging.WARNING, logger='ptycho.params'):
            params.set('intensity_scale', 5.0)   # whitelisted -> NO warn
        assert not any('post-seal' in r.message for r in caplog.records), \
            "whitelisted key must not warn"
    finally:
        params.unseal()
        if had_is:
            params.cfg['intensity_scale'] = prev_is
        else:
            params.cfg.pop('intensity_scale', None)


def test_unsealed_writes_silent(caplog):
    prev = params.get('gridsize', 1)
    try:
        params.unseal()
        with caplog.at_level(logging.WARNING, logger='ptycho.params'):
            params.set('gridsize', 3)            # unsealed -> NO warn
        assert not any('post-seal' in r.message for r in caplog.records), \
            "unsealed writes must not warn"
    finally:
        params.unseal()
        params.set('gridsize', prev)
