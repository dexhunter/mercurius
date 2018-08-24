def start_web():
    from mercurius.web.app import main
    main()

def test_kline():
    symbols = ['ETH/BTC', 'ETC/BTC', 'LTC/BTC', 'XRP/BTC', 'XLM/BTC', 'XEM/BTC']
    #symbols = ['ETH/BTC', 'ETC/BTC', 'LTC/BTC', 'XRP/BTC']
    from mercurius.data.candlereader import CandleReader
    close_data = CandleReader(symbols, '2018-03-24 00:00:00', '2018-04-24 00:00:00', '30m', 'poloniex').get_close(False, True)
    print(close_data)

    from mercurius.strategy.ons import ons
    from mercurius.strategy.ucrp import ucrp
    from mercurius.strategy.ubah import ubah
    from mercurius.strategy.best import best

    #re = olmar.olmar().get_b(close_data, np.ones(2)/2)
    #print(close_data)
    uc = ucrp()
    uc.trade(close_data, tc=0.025)
    re = uc.finish(True, True)
    ub = ubah()
    ub.trade(close_data, tc=0.025)
    re2 = ub.finish(True, True)
    be = best()
    be.trade(close_data, tc=0.025)
    re3 = be.finish(True, True)

    print(re['portfolio'])
    print(re2['portfolio'])
    print(re3['portfolio'])
    print(type(re3['portfolio']))

def test_algo():
    from mercurius.strategy.olmar import olmar
    from mercurius.strategy import tools

    tools.run(olmar(), min_period=2)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

    #test_algo()
    #test_kline()
    start_web()
