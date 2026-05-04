"""Analysis code for the song-vs-music ECoG project (one folder = one package).

Start in ``config`` (paths and numbers), then ``pipeline`` to regenerate
``results/``. Other modules split the work: load data, decode, build RDMs,
run stats, plot. Notebooks often add ``logic/`` to ``sys.path`` and import
flat names like ``import pipeline``.

Rough map: ``config``, ``data_utils``, ``subsets``, ``decoding``, ``rdm``,
``stats``, ``analyses``, ``nonlinear``, ``bellier_data``, ``bellier_decoder``,
``temporal_profile``, ``plots``, ``pipeline``, ``cache_io``.
"""
