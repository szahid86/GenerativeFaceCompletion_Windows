function delta_D = get_delta_D(Solver_D_, Solver_D, batch, gt_D, iter)

trainingLossfile = fopen('../FaceCompletion_training/model/training_loss_results.txt','a');  %Sidra added

    active_D = Solver_D_.net.forward(batch);  
    delta_D = cell(size(active_D));
    for c = 1:length(active_D)
        active_D_ = active_D{c};
        delta_D{c} = zeros(size(active_D{c}));
        [delta_D_, loss_D] = GFC_BCE_loss2(active_D_, gt_D, 'train');
        Solver_D.loss(iter) = loss_D;
%         fprintf('loss_G = %i', loss_D); fprintf('  ');
        fprintf(trainingLossfile, ' loss_G = %4.8f', loss_D); %sidra added
        fclose(trainingLossfile); %sidra added
        delta_D{c} = delta_D_;
    end
end