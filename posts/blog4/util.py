def plot_perceptron_training(X: torch.Tensor, y: torch.Tensor, model, optimizer, k: int = 0, 
                             max_iters: int = 1000, alpha: float =  0.01):

    # Initialize model and optimizer
    p = model()
    opt = optimizer(p)
    p.loss(X, y)

    fig, axarr = plt.subplots(1, 3, figsize=(18, 6))

    loss = 1
    loss_vec = []
    score_vec = []

    iteration = 0

    while loss > 0 and iteration <= max_iters:

        if k == 0:
            i = torch.randint(X.size(0), size=(1,))
            x_i = X[[i],:]
            y_i = y[i]
        else:
            ix = torch.randperm(X.size(0))[:k]
            x_i = X[ix,:]
            y_i = y[ix]

        local_loss = p.loss(x_i, y_i).item()
        score = p.score(X).mean()

        if local_loss > 0:
            if k == 0:
                opt.step(x_i, y_i)
            else:
                opt.step(x_i, y_i, alpha=alpha, mini_batch = True)

            loss = p.loss(X, y).item()
            loss_vec.append(loss)
            score_vec.append(score)

        iteration += 1

    # Plot score over iterations
    axarr[0].plot(range(len(score_vec)), score_vec, color="steelblue", label="Score")
    axarr[0].set_title("Score vs. Iterations")
    axarr[0].set_xlabel("Iteration")
    axarr[0].set_ylabel("Score")

    # Plot loss over iterations
    axarr[1].plot(range(len(loss_vec)), loss_vec, color="steelblue", label="Loss")
    axarr[1].set_title("Loss vs. Iterations")
    axarr[1].set_xlabel("Iteration")
    axarr[1].set_ylabel("Loss")

    # Plot final decision boundary
    plot_perceptron_data(X, y, axarr[2])
    draw_line(p.w, x_min=-1, x_max=2, ax=axarr[2], color="black")
    axarr[2].set_title(f"Final Decision Boundary (Loss = {loss:.3f})")
    axarr[2].set(xlim=(-1, 2), ylim=(-1, 2))

    plt.tight_layout()
    plt.show()
