def summarize_performance(epoch, g_model, dataset, n_samples=3):
    [X1_real, X2_real], _ = generate_real_samples(dataset, n_samples, 1)
    X2_fake, _ = generate_fake_samples(g_model, n_samples, 1)

    # Rescale the images
    X1_real = np.uint8((X1_real + 1) * 127.5)
    X2_real = np.uint8((X1_real + 1) * 127.5)
    X2_fake = np.uint8((X1_real + 1) * 127.5)

    # Plot the images
    for i in range(n_samples):
        plt.subplot(n_samples, 3, i+1)
        plt.title("Real Source Image")
        plt.imshow(X1_real[i])
        plt.axis(False)

        plt.subplot(n_samples, 3, i+1)
        plt.title("Real Target Image")
        plt.imshow(X2_real[i])
        plt.axis(False)

        plt.subplot(n_samples, 3, i+1)
        plt.title("Generated Target Image")
        plt.imshow(X2_fake[i])
        plt.axis(False)
    plt.show()

    # Save the generator model
    fname1 = f"model_{epoch+1}.h5"
    g_model.save(fname1)

    # Save plot as .png file
    fname2 = f"plot_{epoch+1}.png"
    plt.savefig(fname2)

    print(f"Saving Model: {fname1} and Plot: {fname2}...")
    plt.close()

    
def plot_images(src_img, gen_img, tar_img):
    images = np.vstack([src_img, gen_img, tar_img])
    images = (images + 1) * 127.5 # Rescale the images
    
    titles = ["Source Image", "Generated Image", "Image Expected"]
    
    # Plot the images
    for i in range(len(images)):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis(False)
    plt.show()
