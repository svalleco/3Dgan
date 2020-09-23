# THIS IS NOT A FUNCTIONAL FILE, IT IS JUST TO REFERENCE HOW THE ANGLEGAN IS TRAINED FOR THE PGAN IMPLEMENTATION

# Training Function - build & compile discriminator, build & compile generator, run the generator and discriminator, unused callback list functions, read TrainFiles & TestFiles,
#                     run through epochs, train the generator & discriminator, collect discriminator losses, collect generator losses, test, save weights every epoch
def Gan3DTrainAngle(discriminator, generator, opt, datapath, nEvents, WeightsDir, pklfile, global_batch_size, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], warmup_epochs=0):
    start_init = time.time()
    verbose = False    
    particle='Ele'
    f = [0.9, 0.1]
    loss_ftn = hist_count
    
    if hvd.rank()==0:
        print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    if hvd.rank()==0:
        print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=opt,
        loss='binary_crossentropy'
    )
 
    # build combined Model
    # generator: latent vector --> fake image
    latent = Input(shape=(latent_size, ), name='combined_z')   # random latent vector = generator input
    fake_image = generator( latent)     # fake image = generator output
     # discriminator: fake image --> fake, aux, ang, ecal, add_loss
    discriminator.trainable = False
    fake, aux, ang, ecal, add_loss= discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ang, ecal, add_loss],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )
    if kv2: 
        discriminator.trainable = True #workaround for keras 2 bug
        
    gcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    dcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    ccb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    gcb.set_model( generator )
    dcb.set_model( discriminator )
    ccb.set_model( combined )

    gcb.on_train_begin()
    dcb.on_train_begin()
    ccb.on_train_begin()

    # Getting Data
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    if hvd.rank()==0:
        print(Trainfiles)
        print(Testfiles)
    nb_Test = int(nEvents * f[1]) # The number of test files calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train files calculated from fraction of nEvents
    
    # Bug check for reading the test file in
    if len(Testfiles) == 0:
       print('Error reading the Testfiles. The enumerated list will show up as empty. Check the GANutils.py file in 3Dgan/keras/analysis/utils.')
       
    # Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
       else:
           if X_test.shape[0] < nb_Test:
              X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ang_test = np.concatenate((ang_test, ang_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    if X_test.shape[0] > nb_Test:
        X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
    else:
        nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents
    
    # Read train data into a single array, make sure it is the same length as nb_Train (The number of train files calculated from fraction of nEvents)
    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            X_train, Y_train, ang_train, ecal_train = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
        else:
            X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ang_train = np.concatenate((ang_train, ang_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    nb_train = X_train.shape[0]    # Total events in training files
    total_batches = nb_train / global_batch_size
    
    if hvd.rank()==0:
        print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    if hvd.rank()==0:           # will throw an error if the number of epochs is not large enough
       print('Test Data loaded of shapes:')
       print(X_test.shape)
       print(Y_test.shape)
       print('*************************************************************************************')
       print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_test), np.amax(ang_test), np.mean(ang_test)))
       print('Cell varies from {} to {} with mean {}'.format(np.amin(X_test[X_test>0]), np.amax(X_test[X_test>0]), np.mean(X_test[X_test>0])))
       
       if analyse:
          var = gan.sortEnergy(X_test, Y_test, ang_test, ecal_test, energies)
       train_history = defaultdict(list)
       test_history = defaultdict(list)
       analysis_history = defaultdict(list)
       init_time = time.time()- start_init
       print('Initialization time is {} seconds'.format(init_time))
    
    # run through epochs
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        if hvd.rank()==0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
 
        epoch_gen_loss = []
        epoch_disc_loss = []
        randomize(X_train, Y_train, ecal_train, ang_train)

        epoch_gen_loss = []
        epoch_disc_loss = []
        
        image_batches = genbatches(X_train, batch_size)    # creates len(X_train) index ranges for len(batch_size) # of items
        energy_batches = genbatches(Y_train, batch_size)   # creates len(Y_train) index ranges for len(batch_size) # of items
        ecal_batches = genbatches(ecal_train, batch_size)  # creates len(ecal_train) index ranges for len(batch_size) # of items
        ang_batches = genbatches(ang_train, batch_size)    # creates len(ang_train) index ranges for len(batch_size) # of items
        
         # go through batches: train the generator and discriminator
         for index in range(int(total_batches)):
            start = time.time()         
            image_batch = next(image_batches) 
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)
            ang_batch = next(ang_batches)
            add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower), axis=-1)
            noise = np.random.normal(0, 1, (batch_size, latent_size-2))
            generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
  
            # collect the loss of the discriminator with real and fake images
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])

            # if ecal sum has 100% loss then end the training
            if fake_batch_loss[4] == 100.0 and index >10:
                if hvd.rank()==0:
                    print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                    generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                    discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                    print ('real_batch_loss', real_batch_loss)
                    print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])
            trick = np.ones(batch_size)
            
            # collect generator losses in array
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size-1))
                generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch, add_loss_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
            #print ('generator_loss', generator_loss)
            index +=1

            # Used at design time for debugging
            #print('real_batch_loss', real_batch_loss)
            #print ('fake_batch_loss', fake_batch_loss)
            #disc_out = discriminator.predict(image_batch)
            #print('disc_out')
            #print(np.transpose(disc_out[4][:5].astype(int)))
            #print('add_loss_batch')
            #print(np.transpose(add_loss_batch[:5]))

        # Testing  
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        if hvd.rank()==0:
            if analyse:
                result = gan.OptAnalysisShort(var, generated_images, energies)
                print('Analysing............')
                analysis_history['total'].append(result[0])
                analysis_history['energy'].append(result[1])
                analysis_history['moment'].append(result[2])
                analysis_history['angle'].append(result[3])
                print('Result = ', result)
                pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

            print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s} | {6:8s}'.format('component', *discriminator.metrics_names))
            print('-' * 65)
            ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}| {6:<10.2f}'
            print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))

            # save weights every epoch
            generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
            discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)
        
            epoch_time = time.time()-test_start
            pickle.dump({'train': train_history}, open(pklfile, 'wb'))

