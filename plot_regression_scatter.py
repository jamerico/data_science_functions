def plot_regression_scatter(train_results, test_results, title = '', metrics = ['RMSE', 'MAE', 'R2'], 
                            train_title = 'Train', test_title = 'Test', xlabel = 'True', ylabel = 'Predicted', 
                            train_legend_prefix = 'Train results', test_legend_prefix = 'Test results', 
                            save_path = None):
    fig, ax = plt.subplots(ncols = 2, figsize = (17,8))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if title:
        fig.suptitle(title)

    data_min = min(train_results["true"].min(), train_results["predicted"].min(), 
                   test_results["true"].min(), test_results["predicted"].min())
    data_max = max(train_results["true"].max(), train_results["predicted"].max(), 
                   test_results["true"].max(), test_results["predicted"].max())
    data_range = data_max - data_min
    plot_lim = (data_min - 0.05*data_range, data_max + 0.05*data_range)

    train_rmse = round(root_mean_squared_error(train_results["true"], train_results["predicted"]), 5)
    train_mae = round(mean_absolute_error(train_results["true"], train_results["predicted"]), 5)
    train_r2 = round(r2_score(train_results["true"], train_results["predicted"]), 5)
    train_results_string = f'{train_legend_prefix}:'
    if 'RMSE' in metrics:
        train_results_string = f'{train_results_string}\nRMSE:       {train_rmse}'
    if 'MAE' in metrics:
        train_results_string = f'{train_results_string}\nMAE:         {train_mae}'
    if 'R2' in metrics:
        train_results_string = f'{train_results_string}\nR2 Score:  {train_r2}'

    ax[0].grid()
    ax[0].set_xlim(plot_lim)
    ax[0].set_ylim(plot_lim)
    ax[0].plot(plot_lim, plot_lim, 'k-.')
    ax[0].set_title(train_title)
    ax[0].text(0.05, 0.95, train_results_string, transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    sns.regplot(x = 'true', y = 'predicted', data = train_results, ax = ax[0])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)


    test_rmse = round(root_mean_squared_error(test_results["true"], test_results["predicted"]), 5)
    test_mae = round(mean_absolute_error(test_results["true"], test_results["predicted"]), 5)
    test_r2 = round(r2_score(test_results["true"], test_results["predicted"]), 5)
    # test_results_string = f'{test_legend_prefix}:\nRMSE:      {test_rmse}\nMAE:        {test_mae}\nR2 Score: {test_r2}'
    test_results_string = f'{test_legend_prefix}:'
    if 'RMSE' in metrics:
        test_results_string = f'{test_results_string}\nRMSE:       {test_rmse}'
    if 'MAE' in metrics:
        test_results_string = f'{test_results_string}\nMAE:         {test_mae}'
    if 'R2' in metrics:
        test_results_string = f'{test_results_string}\nR2 Score:  {test_r2}'

    ax[1].grid()
    ax[1].set_xlim(plot_lim)
    ax[1].set_ylim(plot_lim)
    ax[1].plot(plot_lim, plot_lim, 'k-.')
    ax[1].set_title(test_title)
    ax[1].text(0.05, 0.95, test_results_string, transform=ax[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    sns.regplot(x = 'true', y = 'predicted', data = test_results, ax = ax[1])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)

    if save_path:
        fig.patch.set_alpha(0.0)
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight', pad_inches = 0)