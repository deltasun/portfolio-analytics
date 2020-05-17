# https://cran.r-project.org/web/packages/PerformanceAnalytics/vignettes/portfolio_returns.pdf
import pandas as pd
import numpy as np
import warnings


class Portfolio(object):
    """

    """

    def __init__(self, prices):
        """

        :param prices: pd.DataFrame with price time series in columns
        """

        assert isinstance(prices, pd.DataFrame)
        self.prices = prices

    def compute_returns(self, method="simple"):
        """

        :param method:
        :param type:
        :return:
        """

        def ret_fun(x, method):
            if method == "simple":
                return x / x.shift(1, fill_value=x[0]) - 1
            elif method == "log":
                return np.log(x / x.shift(1, fill_value=x[0]))
            else:
                raise ValueError("method should be either simple or log")

        return self.prices.apply(lambda x: ret_fun(x, method), axis=0)

    @staticmethod
    def get_components_value_single_period(ret, v_init, method="simple"):
        """
        compute components values over time, in a single rebalancing window, given returns and initial values
        :param method:
        :param ret: pd.DataFrame, with .index dates and containing components returns over time
        :param v_init: initial components values
        :return:
        """

        if isinstance(v_init, pd.Series):
            v_init = [v_init.values.tolist()]
        elif isinstance(v_init, pd.DataFrame):
            v_init = v_init.values.tolist()
        else:
            raise ValueError("v_init should be either pd.Series or pd.DataFrame")

        components_value = pd.DataFrame(v_init * ret.shape[0], index=ret.index, columns=ret.columns)
        if method == "simple":
            components_value = ret.apply(lambda x: np.cumprod(1 + x), axis=0) * components_value
        elif method == "log":
            components_value = ret.apply(lambda x: np.cumsum(x), axis=0) * components_value
        else:
            raise ValueError("method should be either simple or log")

        return components_value

    def portfolio_returns(self, method="simple", weights=None, V0=100, leverage=1, verbose=False):
        """

        :param method:
        :param weights: if None, assume a buy-hold equally weighted ptf. otherwise pd.DataFrame, with .index rebalancing dates
        :param V0: float, initial portfolio value
        :param leverage: float, the maximum investment. if sum(w) > leverage, then rebase to leverage.
                            if sum(w) < leverage, then create residual weight with zero returns.
                            if None, do not adjust weights.
        :param verbose: if True, returns components contributions to portfolio returns
        :return: portfolio returns. if verbose=True, return tuple with ptf rets, contribs
        """

        returns = self.compute_returns(method)

        if weights is None:
            N = returns.shape[1]
            weights = pd.DataFrame([np.repeat(1 / N, N)], index=[returns.index[0]], columns=returns.columns)

        if leverage is not None:
            if any(weights.sum(axis=1) > leverage):
                warnings.warn("sum of weights exceed leverage value of {} in dates {}:\nrebasing to {}".format(
                    leverage, weights[weights.sum(axis=1) > leverage].index.values, leverage))
                weights[weights.sum(axis=1) > leverage] = weights[weights.sum(axis=1) > leverage].apply(
                    lambda x: x / sum(x) * leverage, axis=1
                )

            if not all(np.isclose(weights.sum(axis=1), leverage, rtol=1e-09)):
                warnings.warn(
                    "one or more rebalancing dates have weights not summing up to 1:\nadd a residual weight to compensate")
                weights["residual"] = leverage - weights.sum(axis=1)
                returns["residual"] = 0


        # subset returns to match weights.columns
        returns = returns[weights.columns.tolist()]
        # subset weights to be inside returns dates
        idx = [ind for ind in weights.index if ind in returns.index[:-1]]
        if idx != weights.index.to_list():
            warnings.warn("Some rebalancing dates don't match prices dates. Non matching dates will not be considered.")
            weights = weights.loc[idx]

        V_bop = list()
        V = list()
        n_iter = len(weights.index)

        for t in range(n_iter):
            if t == 0:  # first rebalancing date,
                # get the values of each component at first rebalancing date
                v_bop = V0 * weights.iloc[t]
            else:
                # not the first rebal date, set v_init equal to last available V
                v_bop = V[-1].tail(1).sum(axis=1).values * weights.iloc[t]

            V_bop.append(v_bop.to_frame().transpose())

            # subset returns
            if t != n_iter - 1:
                tmp_ret = returns.loc[weights.index[t]:weights.index[t + 1]]
            else:
                # se è l'ultima iterazione prendi i ritorni fino all'ultima data disponibile
                tmp_ret = returns.loc[weights.index[t]:]

            # notice that subsetting by index includes both extremes!
            # we need to remove the first return, since rebalancing happens from the day after
            # the actual index indicated in the weights input
            tmp_ret = tmp_ret.drop(index=weights.index[t])
            # cumulate returns components inside this interval, i.e. in
            # (index[t] + 1, index[t+1]]
            tmp_ret = self.get_components_value_single_period(tmp_ret, v_bop, method)
            # append values both to V_bop and to V
            # to V_bop we attach not the last value, since the last bop will
            # be replaced by the new v_bop
            V_bop.append(tmp_ret.iloc[:-1])
            V.append(tmp_ret)

        # concat results to get the full components values over time

        # we attach to V the first element
        # corresponding to the first V_bop,
        # notice that this is a bit fictitious, since
        # the eop of the very first rebalancing day is not known,
        # we only know the bop of the day after the rebalancing day
        V.insert(0, V_bop[0])
        V = pd.concat(V)
        # here we need to attach an even more fictitious term,
        # the bop of the first rebalancing day,
        # this is done only for index compatibility with V, it does not matter
        V_bop.insert(0, V_bop[0])
        V_bop = pd.concat(V_bop)
        # assign index to values, index starts at the first date of rebalancing
        V.index = returns.loc[weights.index[0]:].index
        V_bop.index = returns.loc[weights.index[0]:].index

        # portfolio timeseries
        ptf = V.sum(axis=1)
        # portfolio returns
        if method == "simple":
            ptf_ret = ptf / ptf.shift(1) - 1
        elif method == "log":
            ptf_ret = np.log(ptf / ptf(1))
        else:
            raise ValueError("method should be either simple or log")
        # remove first return which is NaN
        ptf_ret.dropna(inplace=True)

        if verbose:
            # compute components' contributions in each day via
            # contrib_i = V_i - Vbop_i / sum(Vbop)
            contrib = V.add(-V_bop).divide(V_bop.sum(axis=1), axis=0)
            return ptf_ret, ptf, contrib, V, V_bop

        return ptf_ret, ptf


