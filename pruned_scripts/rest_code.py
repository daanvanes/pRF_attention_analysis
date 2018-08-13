
#########################################################################################################
# FROM VISUAL_FIELD_PLOTS:
#########################################################################################################



if movement_plot:

	f=pl.figure(figsize=(6,6))
	with sn.axes_style("dark"):
		s=f.add_subplot(111)
		n_bins = 16
		bin_range = np.linspace(-1,1,n_bins,endpoint=True)
		color_wheel = np.zeros((n_bins,n_bins))
		for bx in range(len(bin_range)):
			for by in range(len(bin_range)):
				angle = np.degrees(np.arctan2(bin_range[by],bin_range[bx])+np.pi)
				if angle < 0:
					angle = 360+angle
				color_wheel[by,bx] = angle

			im=pl.imshow(color_wheel,interpolation='nearest',origin='lowerleft',cmap='hsv')

		pl.axhline((n_bins-1)/2,linestyle='--',color='w')
		pl.axvline((n_bins-1)/2,linestyle='--',color='w')

	pl.tight_layout()
	pl.savefig(os.path.join(self.group_plot_dir,'polar_plot','color_wheel.pdf'))
			
	for ci, comparison in enumerate(comparisons.keys()):
		
		print 'creating movement plot for comparison %s...'%comparison

		f = pl.figure(figsize=(24,3))
		sub_k=0

		for subject in these_subjects:
			for ri,roi in enumerate(self.rois_for_plot):
					
				with sn.axes_style("dark"):

					sub_k +=1
					s = f.add_subplot(len(these_subjects),len(self.rois_for_plot),sub_k)
					# pl.title('%s %s'%(subject,roi))
					pl.title('%s'%(roi))

					# We'l create a mask where ecc from both conditions is below self.ecc_threshold[1], 
					# and where the r_squared is higher than self.r_squared_treshold in both conditions
					mask_cond_0 = self.create_mask(np.array(self.all_results[subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']],
							np.squeeze(self.all_stats[subject][comparisons[comparison][0]][roi]))
					mask_cond_1 = self.create_mask(np.array(self.all_results[subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']],
							np.squeeze(self.all_stats[subject][comparisons[comparison][1]][roi]))
					mask = mask_cond_0 & mask_cond_1

					# we want the weights to be the minimum r_squared from both conditions. This makes sense,
					# because a difference measure will only be as solid as the minimum r_squared will be. 
					# when one of the two measures makes complete sense, while the other doesn't, the difference 
					# still doesn't make sense. 
					weights = np.min([np.squeeze(self.all_stats[subject][comparisons[comparison][0]][roi]),
						np.squeeze(self.all_stats[subject][comparisons[comparison][1]][roi])],axis=0)[mask]

					x0 =  np.abs(np.array(self.all_results[subject][comparisons[comparison][0]][roi])[mask,self.results_frames['xo']])
					y0 =  np.abs(np.array(self.all_results[subject][comparisons[comparison][0]][roi])[mask,self.results_frames['yo']])
					x1 =  np.abs(np.array(self.all_results[subject][comparisons[comparison][1]][roi])[mask,self.results_frames['xo']])
					y1 =  np.abs(np.array(self.all_results[subject][comparisons[comparison][1]][roi])[mask,self.results_frames['yo']])

					directions = np.zeros((len(bins),len(bins)))
					for bx in range(len(bins)):
						for by in range(len(bins)):
							these_voxels = (x1>bins[bx,0])*(x1<bins[bx,1])*(y1>bins[by,0])*(y1<bins[by,1])
							# these_voxels *= (these_voxels<np.mean(these_voxels)+np.std(these_voxels)*4))
							# these_voxels *= (these_voxels>np.mean(these_voxels)-np.std(these_voxels)*4))
							# x_fix = np.average(x1[these_voxels],weights=weights[these_voxels])
							# y_fix = np.average(y1[these_voxels],weights=weights[these_voxels])

							x_diff_temp = (x0-x1)
							y_diff_temp = (y0-y1)
							mean_x = np.mean(x0-x1)
							mean_y = np.mean(y0-y1)
							std_x = np.std(x0-x1)
							std_y = np.std(y0-y1)
							mean_angle = np.degrees(np.arctan2(mean_y,mean_x))
							std_angle = np.degrees(np.arctan2(std_y,std_x))
							angle_temp = np.degrees(np.arctan2(y_diff_temp, x_diff_temp))		
							# these_voxels *= (angle_temp < (mean_angle+std_angle*3) )		
							# these_voxels *= (angle_temp > (mean_angle-std_angle*3) )					
		
							try:
								x_diff = np.average((x0-x1)[these_voxels],weights=weights[these_voxels])
								y_diff = np.average((y0-y1)[these_voxels],weights=weights[these_voxels])
								angle = np.degrees(np.arctan2(y_diff, x_diff))
							except:
								angle = np.nan

							# if angle < 0:
								# angle = 360+angle
							# if not np.isnan(angle):
								# pl.imshow(self.two_d_gauss(x_fix,y_fix,np.linalg.norm([x_fix,y_fix])))#,color=colors[int(np.round(angle))])
								# pl.plot(bx,by,'o',color=colors[int(np.round(angle))],markersize=np.linalg.norm([bx,by])*5,alpha=0.5)
							directions[by,bx] = angle


					# pl.xlim(self.ecc_thresholds[0],self.ecc_thresholds[1])
					# pl.ylim(self.ecc_thresholds[0],self.ecc_thresholds[1])
					im=pl.imshow(directions,interpolation='nearest',origin='lowerleft',cmap='hsv',aspect=1)

					# pl.axhline((n_bins-1)/2,linestyle='--',color='w')
					# pl.axvline((n_bins-1)/2,linestyle='--',color='w')

					pl.xticks([])
					pl.yticks([])
		pl.tight_layout()
		pl.savefig(os.path.join(self.group_plot_dir,'polar_plot','movement_%s_over_xy.pdf'%(comparison)))





if plot_type == 'imshow':

	fix_starts = []
	fix_diffs = []
	bins = np.array([[bin_range[b],bin_range[b+1]] for b in range(len(bin_range)-1)])	

	imshow_values = np.zeros((len(bins),len(bins)))
	for bx in range(len(bins)):
		for by in range(len(bins)):
			these_voxels = (fix_data[0,:]>bins[bx,0])*(fix_data[0,:]<bins[bx,1])*(fix_data[1,:]>bins[by,0])*(fix_data[1,:]<bins[by,1])
				
			try:
				# first, we should get the average stim and fix vector for these voxels
				avg_fix = np.average(fix_data[:,these_voxels],weights=weights[these_voxels],axis=1)
				avg_stim = np.average(stim_data[:,these_voxels],weights=weights[these_voxels],axis=1)
				if measure == 'polar':
					# now we can compute the angle between these two vectors. 
					# The formula for this is: arccos(dot(a,b) / (norm(a)*norm(b)))
					this_angle = np.degrees(np.arccos( np.dot(avg_fix,avg_stim) / (np.linalg.norm(avg_fix) * np.linalg.norm(avg_stim)) ))
					# This angle is still 'unsigned'. It doesn't show whether it is a clockwise or counterclockwise rotation.
					# To get this, we can rotate the fix vector by 90 degrees, and see whether the dotproduct of the stim vector
					# and this rotated fix vector is negative or positive: this indicates the rotation direction.
					# A trick to rotate the vector 90 degrees clockwise is to do : [y,-x]
					rotated_avg_fix = np.array([avg_fix[1],-avg_fix[0]])
					clockwise = np.sign(np.dot(rotated_avg_fix,avg_stim))
					this_value = (this_angle * clockwise)
				elif measure == 'size':
					fix_size = np.average(fix_sizes[these_voxels],weights=weights[these_voxels])
					stim_size = np.average(stim_sizes[these_voxels],weights=weights[these_voxels])
					this_value = np.log(stim_size / fix_size)
				elif measure == 'ecc':
					# now we can compute the difference in the norm of the stim and fix vector to get eccentricity differences
					stim_eccen = np.linalg.norm(avg_stim)
					fix_eccen = np.linalg.norm(avg_fix)
					# this_value = np.log(stim_eccen / fix_eccen)
					this_value = stim_eccen - fix_eccen
				elif measure == 'arrow':
					fix_starts.append(avg_fix)
					fix_diffs.append(avg_stim - avg_fix)
					this_value = np.nan
			except:
				this_value = np.nan

			imshow_values[by,bx] = this_value

	fix_starts = np.array(fix_starts)
	fix_diffs = np.array(fix_diffs)

	if measure != 'arrow':
		# let's get rid of outliers
		imshow_values[(imshow_values>(np.nanmean(imshow_values)+np.nanstd(imshow_values)*self.outlier_num_stds))+(imshow_values<(np.nanmean(imshow_values)-np.nanstd(imshow_values)*self.outlier_num_stds))] = np.nan
		maxval = np.max(np.abs(imshow_values[-np.isnan(imshow_values)]))
		blurred_imshow = blur_image(imshow_values,2)
		im=pl.imshow(imshow_values,interpolation='none',origin='lowerleft',cmap='seismic',aspect=1,vmin=-maxval,vmax=maxval)
		# pl.colorbar()
		if field == 'whole_field':
			pl.axhline((n_bins/2)-0.5,linestyle='--',color='w')
			pl.axvline((n_bins/2)-0.5,linestyle='--',color='w')
			pl.xticks(np.linspace(0,n_bins-0.5,7),np.linspace(-self.stim_radius,self.stim_radius,7))
			pl.yticks(np.linspace(0,n_bins-0.5,7),np.linspace(-self.stim_radius,self.stim_radius,7))
		elif field == 'quadrant':		
			pl.xticks(np.linspace(0,n_bins-0.5,4),np.linspace(0,self.stim_radius,4))
			pl.yticks(np.linspace(0,n_bins-0.5,4),np.linspace(0,self.stim_radius,4))
	else:
		for bi in range(np.shape(fix_starts)[0]):
			pl.arrow(fix_starts[bi,0],fix_starts[bi,1],fix_diffs[bi,0],fix_diffs[bi,1],color='k',head_width=0.03,width=0.003	)
		if field == 'whole_field':
			pl.xlim(-1,1)
			pl.ylim(-1,1)
			pl.axhline(0,linestyle='--',color='w')
			pl.axvline(0,linestyle='--',color='w')
			pl.xticks(np.linspace(-1,1,7),np.linspace(-self.stim_radius,self.stim_radius,7))
			pl.yticks(np.linspace(-1,1,7),np.linspace(-self.stim_radius,self.stim_radius,7))
		elif field == 'quadrant':
			pl.xlim(0,1)
			pl.ylim(0,1)			
			pl.xticks(np.linspace(0,1,4),np.linspace(0,self.stim_radius,4))
			pl.yticks(np.linspace(0,1,4),np.linspace(0,self.stim_radius,4))



#########################################################################################################
# SPATIAL NORMALIZATION CODE
#########################################################################################################
	def unwarp_EPIs(self,fnirt_postFix=['mcf','reg'],applywarp_postFix = ['mcf','reg'],target_session=-1):

		"""
		This function should be run after motion correction to the session T2 has been run.
		It takes the mean motion corrected volume and warps it nonlinearly (using FNIRT) to the session T2.
		This warpfield will then be applied to the original, non-motion corrected nifti. 
		This way, B0 distortion will be corrected for. Motion correction and between session registration
		now still need to be run on the _warped file to the session T2. 
		"""

		fnirts = []
		warps = []

		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

			# first, we'll configure a FNIRT to the mean motion corrected volume
			input_fn = self.runFile(stage = 'processed/mri', run = r, postFix = fnirt_postFix + ['meanvol'])
			output_fn = self.runFile(stage = 'processed/mri', run = r, postFix = fnirt_postFix + ['meanvol','warped'])
			# target_T2 = self.runFile(stage='processed/mri/T2_anat/'+str(r.mocoT2anatIDtarget)+'/', postFix = [str(r.mocoT2anatIDtarget)])
			target_T2 = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]])
			coefs_fn = self.runFile(stage = 'processed/mri', run = r, postFix = fnirt_postFix + ['meanvol','coefs'])

			flO = FnirtOperator(inputObject = input_fn, referenceFileName = target_T2)
			flO.configure(outputFileName=output_fn,coefsFileName=coefs_fn)
			fnirts.append(flO)

			# then, we'll configure an applywarp, applying the resulting warpfield (coefs) to the original, non motion corrected volume
			input_fn = self.runFile(stage = 'processed/mri', run = r, postFix = applywarp_postFix)
			output_fn = self.runFile(stage = 'processed/mri', run = r, postFix = applywarp_postFix + ['warped'])

			awO = ApplyWarpOperator(inputObject = input_fn, referenceFileName = input_fn)
			awO.configure(warpfieldFileName = coefs_fn,outputFileName=output_fn)
			warps.append(awO)

		# execute the fnirts first
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fnirt.runcmd,),(),('subprocess','tempfile',)) for fnirt in fnirts]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()	

		# and then apply the warps to the original epis
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(warp.runcmd,),(),('subprocess','tempfile',)) for warp in warps]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()	

	def coregister_sessions_T2(self):

		"""
		This function checks whether a target session needs to be registered to another session,
		and creates the transformation matrix required to transform one T2 anatomical to the other.
		NB this function DOES NOT perform any transformation on the epis itself. This is done in 
		combine_between_within_session_registration. 
		"""

		for r in [self.runList[i] for i in self.conditionDict['T2_anat']]:


			this_run_T2 = self.runFile(stage='processed/mri/T2_anat/'+str(r.ID)+'/', postFix = [str(r.ID)] )
			target_T2 = self.runFile(stage='processed/mri/T2_anat/'+str(r.targetSessionT2anatID)+'/', postFix = [str(r.targetSessionT2anatID)] )
			output_mat = this_run_T2[:-7] + '_to_%s_NB.mat'%r.targetSessionT2anatID

			if (r.ID != r.targetSessionT2anatID):

				# bet images of they aren't already
				for this_T2 in [this_run_T2,target_T2]:
					if not os.path.isfile(this_T2[:-7] + '_NB.nii.gz'):
						better = BETOperator( inputObject = self.referenceFunctionalFileName )
						self.referenceFunctionalFileName = self.referenceFunctionalFileName[:-7] + '_NB.nii.gz'
						better.configure( outputFileName = self.referenceFunctionalFileName )
						better.execute()

				# now perform FLIRT if it hasn't been done already:
				if not os.path.isfile(output_mat):
					flO = FlirtOperator(inputObject = (this_run_T2[:-7] + '_NB.nii.gz'), referenceFileName = (target_T2[:-7] + '_NB.nii.gz'))
					flO.configureApply()
					flO.configureRun(transformMatrixFileName = output_mat)
					flO.execute()
			else:
				# for the T2 which is the target T2, create an identity matrix
				identity_matrix = np.eye(4)
				np.savetxt(output_mat,identity_matrix,fmt='%.6f')

	def FLIRT_and_FNIRT_to_target_T2(self,target_session=-1,postFix = ['mcf']):

		flirts = []
		fnirts = []
		warps = []

		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

			# first flirt mean mcf volume to target T2 
			flirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]], postFix=['NB'])
			in_meanvol_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'])
			flirt_out_mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'], extension='.mat')
			flirt_out_fn =  self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol','trans'])

			flO = FlirtOperator(inputObject = in_meanvol_fn, referenceFileName = flirt_target_fn)
			# flO.configureApply()
			flO.configureRun(transformMatrixFileName = flirt_out_mat_fn)
			flirts.append(flO)

			# shell()
			# # then use this matrix to initialize a FNIRT to the target EPI
			fnirt_out_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol','warped'])
			# fnirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]])
			fnirt_target_fn = self.runFile(stage = 'processed/mri', run = self.runList[self.scanTypeDict['epi_bold'][target_session]], postFix=['mcf','meanvol','trans'])
			warp_coefs_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol','warpcoef'])

			# fnO = FnirtOperator(inputObject = in_meanvol_fn, referenceFileName = fnirt_target_fn)
			# fnO.configure(AffineTransMatrixFileName=flirt_out_mat_fn,outputFileName=fnirt_out_fn,coefsFileName=warp_coefs_fn)
			# fnirts.append(fnO)

			fnO = FnirtOperator(inputObject = flirt_out_fn, referenceFileName = fnirt_target_fn)
			fnO.configure(outputFileName=fnirt_out_fn,coefsFileName=warp_coefs_fn)
			fnirts.append(fnO)

			# # then, we'll configure an applywarp, applying the resulting warpfield (coefs) to the original, motion corrected volume
			# warp_input_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix)
			# warp_output_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['warped'])

			# awO = ApplyWarpOperator(inputObject = warp_input_fn, referenceFileName = warp_input_fn)
			# awO.configure(warpfieldFileName = warp_coefs_fn, outputFileName = warp_output_fn)
			# warps.append(awO)

		# execute the flirts first
		# ppservers = ()
		# job_server = pp.Server(ppservers=ppservers, secret='mc')
		# self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		# ppResults = [job_server.submit(ExecCommandLine,(flirt.runcmd,),(),('subprocess','tempfile',)) for flirt in flirts]
		# for fMcf in ppResults:
		# 	fMcf()
		# job_server.print_stats()	

		# # now convert the hexidecimal transformation matrices to decimal format
		# for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
		# 	flirt_out_mat = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'], extension='.mat'))
		# 	np.savetxt(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'], extension='.mat'),flirt_out_mat)

		# then the fnirts
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(fnirt.runcmd,),(),('subprocess','tempfile',)) for fnirt in fnirts]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()

		# # and then apply the warps to the original epis
		# ppservers = ()
		# job_server = pp.Server(ppservers=ppservers, secret='mc')
		# self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		# ppResults = [job_server.submit(ExecCommandLine,(warp.runcmd,),(),('subprocess','tempfile',)) for warp in warps]
		# for fMcf in ppResults:
		# 	fMcf()
		# job_server.print_stats()

		# # fix headers:
		# cmds = []
		# for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
		# 	nii_file_orig = NiftiImage(self.runFile(stage = 'processed/mri', run = r ))
		# 	nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['warped'] ))
		# 	nii_file.header = nii_file_orig.header
		# 	nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['warped'] ))

	def create_mean_vol(self,postFix=['mcf','warped']):

		for r in [self.runList[i] for i in self.conditionDict['PRF']]:

			filename = self.runFile(stage = 'processed/mri', run = r, postFix=postFix )
			whole_data = NiftiImage(filename).data
			mean_data = np.mean(whole_data,axis=0)

			mean_nifti = NiftiImage(mean_data)
			mean_nifti.header = NiftiImage(filename).header
			save_postFix = postFix + ['meanvol']
			save_filename = self.runFile(stage = 'processed/mri', run = r, postFix=save_postFix )
			mean_nifti.save(save_filename)

	def FLIRT_to_target_T2(self,target_session=-1,postFix = ['mcf']):

		flirts = []
		XFM4DOperatorList = []

		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

			# first flirt mean mcf volume to target T2 
			flirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]], postFix=['NB'])
			in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'])
			flirt_out_mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'], extension='.mat')

			if not os.path.isfile(flirt_out_mat_fn): 
				flO = FlirtOperator(inputObject = in_fn, referenceFileName = flirt_target_fn)
				flO.configureApply()
				flO.configureRun(transformMatrixFileName = flirt_out_mat_fn)
				flirts.append(flO)

			in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = postFix )
			ofn = self.runFile(stage = 'processed/mri', run = r, postFix= postFix + ['flirted'])
			if not os.path.isfile(ofn):
				XFM4DO = XFM4DOperator(inputObject = in_fn)
				XFM4DO.configure(transformMatrixFile=flirt_out_mat_fn, regFile = in_fn, outputFileName = ofn)
				XFM4DOperatorList.append(XFM4DO)

		if flirts != []:
			# execute the flirts if files don't already exist
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers, secret='mc')
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(flirt.runcmd,),(),('subprocess','tempfile',)) for flirt in flirts]
			for fMcf in ppResults:
				fMcf()
			job_server.print_stats()

		if not XFM4DOperatorList == []:
			# execute the commands in parallel
			ppservers = ()
			job_server = pp.Server(ppservers=ppservers, secret='mc')
			self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
			ppResults = [job_server.submit(ExecCommandLine,(XFM4D.runcmd,),(),('subprocess','tempfile',)) for XFM4D in XFM4DOperatorList]
			for fMcf in ppResults:
				fMcf()
			
			job_server.print_stats()

		# fix headers:
		cmds = []
		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
			nii_file_orig = NiftiImage(self.runFile(stage = 'processed/mri', run = r ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['flirted'] ))
			nii_file.header = nii_file_orig.header
			nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['flirted'] ))
			
def nuisance_GLM_one_slice(sl,TR,n_TRs,cortex_mask,design_regressors,hrf_parameters,other_regressors,stim_prediction,data,n_slices,plot,plotdir,roi_names,hrf_type):

	slice_mask = np.zeros_like(cortex_mask)
	slice_mask[sl,:,:] = cortex_mask[sl,:,:]	

	residuals = []

	for voxno in range(slice_mask.sum()):

		if hrf_type != 'canonical':
			hrf_params = hrf_parameters[:,slice_mask][:,voxno]
		else:
			hrf_params = [1,0,0]


		# create design matrix
		design =  NewDesign(n_TRs, TR, sample_duration = TR/float(n_slices))
		design.configure(design_regressors, 
			hrf_parameters = hrf_params)

		# if no stim:
		# joined_design_matrix = other_regressors.T
		# if stim:
		# joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors]).T
		# both cases no orth:
		# joined_design_matrix_with_basleine = np.hstack([np.ones((n_TRs,1)),joined_design_matrix])
		# dm = np.mat(joined_design_matrix_with_basleine)

		# # if orth:
		# joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors]).T
		# orthonormal_dm = np.linalg.qr(joined_design_matrix)[0]
		# orthonormal_dm_with_baseline = np.hstack([np.ones((n_TRs,1)),orthonormal_dm])
		# dm = np.mat(orthonormal_dm_with_baseline)

		# before:

		joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors,design.convolved_design_matrix[:,sl::n_slices]]).T
		orthonormal_dm = np.linalg.qr(joined_design_matrix)[0]
		orthonormal_dm_with_baseline = np.hstack([np.ones((n_TRs,1)),orthonormal_dm])
		dm = np.mat(orthonormal_dm_with_baseline)

		these_voxels = data[:,slice_mask][:,voxno]

		# do actual glm
		these_betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(these_voxels.T).T
		all_predictions = np.squeeze([dm[:,regidx] * these_betas[regidx] for regidx in range(np.shape(dm)[1])])
		nuisances = np.ones(np.shape(dm)[1]).astype(bool)
		nuisances[1] = False
		these_nuisances = (np.mat(dm[:,nuisances]) * np.mat(these_betas[nuisances,:])) # subtract all but the stimulus signal
		these_residuals = np.ravel(these_voxels - np.squeeze(these_nuisances))
		residuals.append((these_residuals / np.std(these_residuals)))

		if np.sum(these_voxels) > 0:
			rss_no_glm = np.sum(((these_voxels-all_predictions[0])-all_predictions[1])**2)
			r_squared_no_glm = 1 - rss_no_glm/np.sum(((these_voxels-all_predictions[0]) - np.mean((these_voxels-all_predictions[0]))) ** 2) 
			rss_with_glm = np.sum((these_residuals-all_predictions[1])**2)
			r_squared_with_glm = 1 - rss_with_glm/np.sum((these_residuals - np.mean(these_residuals)) ** 2) 
		
			if plot * (r_squared_no_glm > 0.1) * ((r_squared_with_glm - r_squared_no_glm)>0.1):# * (np.random.rand() < 0.1):
			
				this_plot_dir = os.path.join(plotdir, '%s'%roi_names[slice_mask][voxno])
				if not os.path.isdir(this_plot_dir): os.mkdir(this_plot_dir)

				f = pl.figure(figsize=(24,16))
				s = f.add_subplot(7,1,1)
				pl.plot(these_voxels,label='data')
				pl.plot(these_nuisances,label='nuisances')
				pl.legend(loc='best')
				s = f.add_subplot(7,1,2)
				pl.plot(these_voxels-all_predictions[0],label='data, r_squared: %.2f'%r_squared_no_glm)
				pl.plot(these_residuals,label='residuals, r_squared: %.2f'%r_squared_with_glm)
				pl.plot(all_predictions[1],label='stimulus')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,5)
				pl.plot(all_predictions[0],label='baseline')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,6)
				pl.plot(np.sum(all_predictions[2:20],axis=0),label='speed correction')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,7)
				pl.plot(all_predictions[20],label='mean GM signal')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,8)
				pl.plot(all_predictions[21],label='mean WM signal')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,9)
				pl.plot(all_predictions[22],label='mean CSF signal')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,10)
				pl.plot(np.sum(all_predictions[23:29],axis=0),label='retroicor card')
				pl.plot(np.sum(all_predictions[29:33],axis=0),label='retroicor resp')
				pl.plot(np.sum(all_predictions[33:57],axis=0),label='retroicor int')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,11)
				pl.plot(all_predictions[57],label='blinks')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,12)
				pl.plot(all_predictions[58],label='button presses L')
				pl.plot(all_predictions[59],label='button presses R')
				pl.legend(loc='best')
				s = f.add_subplot(7,2,13)
				pl.plot(all_predictions[60],label='Fix transients')
				pl.plot(all_predictions[61],label='Color transients')
				pl.plot(all_predictions[62],label='Speed transients')
				pl.legend(loc='best')
				pl.tight_layout()

				pl.savefig(os.path.join(this_plot_dir, 'nuisance_design_slice_%d_vox_%d.pdf'%(sl,voxno)))
				pl.close()

	return residuals

def Mapper_nuisance_GLM(self, mask = 'bet_mask', postFix = ['mcf', 'sgtf'],plot_postFix=['mcf'],condition='PRF',target_session =-1,plot=True):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. 

		At the moment, it incorporates the speed correction parameters 
		and eye blinks. Physio yet needs to be implemented.
		"""

		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask + '.nii.gz')).data, dtype = bool)
		
		for r in [self.runList[i] for i in self.conditionDict[condition]]:

			if plot:
				this_dm_plot_dir = os.path.join(self.runFolder('processed/mri/',run=r),'nuisance_design_matrices')
				if os.path.isdir(this_dm_plot_dir): shutil.rmtree(this_dm_plot_dir)
				os.mkdir(this_dm_plot_dir)

			self.logger.info('loading %s'%self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			## load nifti data
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			nii_data_for_plot = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = plot_postFix )).data
			nii_data = nii_file.data
			if nii_file.rtime > 1000:
				rtime = nii_file.rtime/1000
			elif nii_file.rtime < 0.01:
				rtime = nii_file.rtime * 1000
			else:
				rtime = nii_file.rtime
			n_slices = nii_file.volextent[-1]
			n_timepoints = nii_file.timepoints

			## for mapper, load the stimulus times
			no_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_no_speed.txt'))
			no_color_no_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in no_color_no_speed]
			no_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_yes_speed.txt'))
			no_color_yes_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in no_color_yes_speed]
			yes_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_no_speed.txt'))
			yes_color_no_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in yes_color_no_speed]
			yes_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_yes_speed.txt'))
			yes_color_yes_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in yes_color_yes_speed]
			
			## transients
			self.logger.info('loading transient times')
			transients = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'transient_times.txt'))
			transients_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in transients]

			## speed correction parameters
			self.logger.info('loading speed correction parameters')
			mcf = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' ))
			mcf_dt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_dt.par' ))
			mcf_ddt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_ddt.par' ))
			# sgtf these regressors, because they are estimated from the non-sgtf data
			mcf_sgtf = mcf - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf.T]).T
			mcf_dt_sgtf = mcf_dt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_dt.T]).T
			mcf_ddt_sgtf = mcf_ddt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_ddt.T]).T

			## button presses
			self.logger.info('loading button press times')
			button_L = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_L.txt'))
			button_L_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_L]
			button_R = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_R.txt'))
			button_R_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_R]

			## blinks
			self.logger.info('loading blink times')
			this_blink_events = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'blink_times.txt'))
			blink_times_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in this_blink_events]

			## mean WM / GM / CSF volume 
			self.logger.info('loading GM / WM / CSF')
			GM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_0','resampled'])).data.astype(bool)
			WM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_1','resampled'])).data.astype(bool)
			CSF_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_2','resampled'])).data.astype(bool)

			self.logger.info('computing GM / WM / CSF mean per slice - medianed with 4 neighboring slices')
			# meaning data over voxels
			mean_GM_per_slice = np.array([np.mean(nii_data[:,sl,GM_mask[sl]],axis=1) for sl in range(n_slices)])
			mean_WM_per_slice = np.array([np.mean(nii_data[:,sl,WM_mask[sl]],axis=1) for sl in range(n_slices)])
			mean_CSF_per_slice = np.array([np.mean(nii_data[:,sl,CSF_mask[sl]],axis=1) for sl in range(n_slices)])
			# compute median over current slice +/- 4 neighboring slices
			mean_GM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_GM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			mean_WM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_WM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			mean_CSF_per_slice_sm = np.array([[sp.stats.nanmedian(mean_CSF_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			# fill those slices with zeros that contain nans
			mean_GM_per_slice_sm_nonan = np.array([mean_GM_per_slice_sm[:,sl] if np.isnan(mean_GM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			mean_WM_per_slice_sm_nonan = np.array([mean_WM_per_slice_sm[:,sl] if np.isnan(mean_WM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			mean_CSF_per_slice_sm_nonan = np.array([mean_CSF_per_slice_sm[:,sl] if np.isnan(mean_CSF_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			# subtract mean signal from these regressors, so that they do not start sucking op baseline activation
			mean_GM_per_slice_sm_nonan_demeaned = (mean_GM_per_slice_sm_nonan.T-np.mean(mean_GM_per_slice_sm_nonan,axis=1)).T
			mean_WM_per_slice_sm_nonan_demeaned = (mean_WM_per_slice_sm_nonan.T-np.mean(mean_WM_per_slice_sm_nonan,axis=1)).T
			mean_CSF_per_slice_sm_nonan_demeaned = (mean_CSF_per_slice_sm_nonan.T-np.mean(mean_CSF_per_slice_sm_nonan,axis=1)).T

			## retroicor regressors
			self.logger.info('loading retroicor regressors')
			retroicor_dir = os.path.join(self.runFolder('processed/mri/',run=r),'retroicor')
			retroicor_regressors = np.squeeze(np.array([NiftiImage(os.path.join(retroicor_dir,'retroicorev00%d.nii.gz'%(reg+1))).data if reg < 9 else NiftiImage(os.path.join(retroicor_dir,'retroicorev0%d.nii.gz'%(reg+1))).data for reg in np.arange(34)]))

			## create design matrix
			self.logger.info('creating design matrix')
			run_design = NewDesign(nii_file.timepoints, rtime, sample_duration = rtime/float(n_slices))
			run_design.configure([blink_times_list,button_L_list,button_R_list,no_color_no_speed_list,no_color_yes_speed_list,yes_color_no_speed_list,yes_color_yes_speed_list], 
				# hrf_parameters=[1,0,0])
				hrf_type = 'doubleGamma',hrf_parameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35})
			n_regressors = 63

			regressor_names = ['baseline','blinks','button_L','button_R','Fix_transients',
					'moco_x','moco_y','moco_z','moco_roll','moco_pitch','moco_yaw',
					'moco_dt_x','moco_dt_y','moco_dt_z','moco_dt_roll','moco_dt_pitch','moco_dt_yaw',
					'moco_ddt_x','moco_ddt_y','moco_ddt_z','moco_ddt_roll','moco_ddt_pitch','moco_ddt_yaw',
					'mean_GM','mean_WM','mean_CSF',
					'card1','card2','card3','card4','card5','card6',
					'resp1','resp2','resp3','resp4',
					'int1','int2','int3','int4','int5','int6','int7','int8',
					'int9','int10','int11','int12','int13','int14','int15','int16',
					'int17','int18','int19','int20','int21','int22','int23','int24','bar_stimulus']

			## pre allocate full output variables, 
			betas = np.zeros(([n_regressors]+list(cortex_mask.shape)))
			residuals = np.zeros_like(nii_data)

			for sl in np.arange(n_slices):
				self.logger.info('executing nuisance GLM on slice %d'%sl)		

				joined_design_matrix = np.vstack([np.ones(n_timepoints),mcf_sgtf.T,mcf_dt_sgtf.T,mcf_ddt_sgtf.T,
					mean_GM_per_slice_sm_nonan_demeaned[sl],mean_WM_per_slice_sm_nonan_demeaned[sl],mean_CSF_per_slice_sm_nonan_demeaned[sl],
					retroicor_regressors[:,:,sl],run_design.convolved_design_matrix[:,sl:nii_file.timepoints*n_slices:n_slices]]).T

				slice_mask = np.zeros_like(cortex_mask)
				slice_mask[sl,:,:] = cortex_mask[sl,:,:]

				dm = np.mat(joined_design_matrix)

				these_voxels = nii_data[:,slice_mask]
				these_betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(nii_data[:,slice_mask].T).T
				betas[:,slice_mask] = these_betas
				these_residuals = nii_data[:,slice_mask] - (np.mat(dm[:,1:-4]) * np.mat(these_betas[1:-4,:])) # don't take the first baseline and last 4 stimulus regressors into account when computing residuals
				residuals[:,slice_mask] = these_residuals/np.std(these_residuals) # now the data is in z-scores, with baseline=0 but not mean =0

			self.logger.info('nuisance GLM finished; outputting residuals to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))[-1])		
			res_nii_file = NiftiImage(residuals)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res']))
			
			self.logger.info('nuisance GLM finished; outputting betas to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))[-1])			
			betas_nii_file = NiftiImage(betas)
			betas_nii_file.header = nii_file.header
			betas_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['betas']))

			# delete variables to empty memory
			del(betas)
			del(residuals)

			

	def combine_and_apply_between_and_within_session_registration(self,mcf_postFix=['mcf'],in_postFix=[],out_postFix=['mcf-with-flirt'],target_session=-1):

		"""
		Before running this function, motion correction should be performed with the T2 anatomical of that session as target.
		Additionally, we need the transformation matrices from the between session registrations of the T2s. 
		This function will multiply the individual TR transformation matrices that come out of the motion correction with the 
		between session transformation matrix. It then applies this transformation to all the slices of the original volume. 
		This way, we create a 'reg' file, that is now corrected for within and between session motion. 
		The advantage of this method above performing two seperate transformations (say, first motion correction, then FLIRT between sessions),
		is that we now only perform one transformation, which includes only one interpolation step. This reduces the amount of smoothing
		introduced by linear transformations.

		As matrix multiplication is not commutative, we nee6d to think about which 
		transformation to apply first. In this case, we want to solve the within session motion first, and then 
		register to the other session. This is done by multiplying the motion correction matrices with the 
		between session registration matrix.
		"""
		
		flirts = []
		XFM4DOperatorList = []

		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

			# first flirt mean mcf volume to target epi
			flirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]])
			flirt_target_fn_betted = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB'])
			in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = mcf_postFix + ['meanvol'])
			in_fn_betted = self.runFile(stage = 'processed/mri', run = r, postFix = mcf_postFix + ['meanvol','NB'])
			# as flirt works best with BETTED brains, let's first BET the flirt target
			if not os.path.isfile(flirt_target_fn_betted):
				better = BETOperator( inputObject = flirt_target_fn)
				better.configure( outputFileName = flirt_target_fn_betted)
				better.execute()
			# and also the mean motion corrected volume 
			if not os.path.isfile(in_fn_betted):
				better = BETOperator( inputObject = in_fn)
				better.configure( outputFileName = in_fn_betted)
				better.execute()

			# as we'll need the affine transformation matrix that FLIRT produces, we'll explicitly save it
			flirt_out_mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = mcf_postFix + ['meanvol'], extension='.mat')
			# and add to this run object for easy acces
			r.transformationMatrixFile = flirt_out_mat_fn

			# now create and configure the flirt object with the betted brains
			flO = FlirtOperator(inputObject = in_fn_betted, referenceFileName = flirt_target_fn_betted)
			flO.configureRun(transformMatrixFileName = flirt_out_mat_fn)
			# add it to the flirt command list for later execution
			flirts.append(flO)

		# now that we have the linear transformations from all mean motion corrected volumes to the target T2,
		# we can combine them with the respective motion correction warps for every timepoint in the epis. 
		# this way, we can apply both transformations at once, thereby only resulting in one interpolation operation
		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

			# get the rawest func file to apply to
			func_fn = self.runFile(stage = 'processed/mri', run = r, postFix=in_postFix)

			# Read in the flirt registration matrix
			reg_matrix = np.mat(np.loadtxt(r.transformationMatrixFile))	

			# Now, we'll loop over volumes, read in the corresponding motion correction matrix,
			# multiply this together with the flirt registration and save the corresponding matrix.
			self.logger.info('reading motion correction matrices')
			most_recent_moco_folder = glob.glob(os.path.join(self.runFolder(stage = 'processed/mri/', run = r),'*_%s.mat*'%'_'.join(mcf_postFix)))[0]
			output_mat_dir = os.path.join(self.runFolder(stage = 'processed/mri/', run = r),'combined_registration_matrices')
			if os.path.isdir(output_mat_dir): shutil.rmtree(output_mat_dir)
			os.mkdir(output_mat_dir)
			self.logger.info('combining within and between session registrations')
			for timepoint in range(NiftiImage(func_fn).timepoints):
				mat_name = 'MAT_0000'
				mat_name = mat_name[:-len(str(timepoint))] + str(timepoint)
				this_moco_mat = np.mat(np.loadtxt(os.path.join(most_recent_moco_folder,mat_name)))
				combined_matrix = reg_matrix * this_moco_mat # not that both are matrices, not numpy arrays
				output_mat_name = 'MAT_0' + mat_name[4:]
				np.savetxt(os.path.join(output_mat_dir,output_mat_name),combined_matrix,fmt='%.6f')

			# Now, we can use the applyxfm4d utility to apply all these matrices to the 'raw' nifti.
			ofn = self.runFile(stage = 'processed/mri', run = r, postFix=out_postFix)
			XFM4DO = XFM4DOperator(inputObject = func_fn)
			XFM4DO.configure(transformMatrixDir=output_mat_dir, regFile = func_fn, outputFileName = ofn)
			XFM4DOperatorList.append(XFM4DO)

		# execute the flirts if files don't already exist
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(flirt.runcmd,),(),('subprocess','tempfile',)) for flirt in flirts]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()

		# execute the xfm operations in parallel
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(XFM4D.runcmd,),(),('subprocess','tempfile',)) for XFM4D in XFM4DOperatorList]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()

		# fix headers:
		for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
			nii_file_orig = NiftiImage(self.runFile(stage = 'processed/mri', run = r ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = out_postFix ))
			nii_file.header = nii_file_orig.header
			nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = out_postFix ))



	def check_between_session_registration(self,postFix=['warped','mcf','reg'],single_run_movies=True,compare_2_run_movies=True,compare_all_runs_movie=True):

		"""
		This function creates a movie of the functional files at the middle slice, with the volumes shuffled.
		This can give insight into the success of the within session motion correction.

		It also outputs every possible combination of two sessions, as well as a combination with all sessions.
		This can give more insight into the success of between session motion correction.
		"""

		pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
		plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/'))
		if not os.path.isdir(plotdir): os.mkdir(plotdir)

		for condition in self.run_types:

			# load epis
			all_epis = []
			self.logger.info('loading all %s niftis'%condition)
			for r in [self.runList[i] for i in self.conditionDict[condition]]:

				filename = self.runFile(stage = 'processed/mri', run = r, postFix=postFix )
				exec('nifti_session_%s = (NiftiImage(filename).data)'%r.ID)
				exec('all_epis.append(nifti_session_%s)'%r.ID)

			if single_run_movies:
				# creat single run animation with shuffled TRs
				for r in [self.runList[i] for i in self.conditionDict[condition]]:

					self.logger.info('creating %s animation for run %s'%(condition,r.ID))
					ims = []
					f=pl.figure(figsize=(4,4))
					slice_dim = np.argmin(np.shape(eval('nifti_session_%s'%r.ID)))
					timepoints = np.arange(len(eval('nifti_session_%s'%r.ID)))
					np.random.shuffle(timepoints)
					for t in timepoints:
						s=f.add_subplot(111)
						if slice_dim == 1:
							im=pl.imshow(eval('nifti_session_%s'%r.ID)[t,15,:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
						elif slice_dim == 2:
							im=pl.imshow(eval('nifti_session_%s'%r.ID)[t,:,15,:],origin='lowerleft',interpolation='nearest',cmap='gray')
						elif slice_dim == 3:
							im=pl.imshow(eval('nifti_session_%s'%r.ID)[t,:,:,15],origin='lowerleft',interpolation='nearest',cmap='gray')
						pl.axis('off')
						ims.append([im])
					ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
					mywriter = animation.FFMpegWriter(fps = 10)
					self.logger.info('saving to %s_%s_run_%s_registration_check.mp4'%('_'.join(postFix),condition,r.ID))
					ani.save(os.path.join(plotdir,'%s_%s_run_%s_registration_check.mp4'%('_'.join(postFix),condition,r.ID)),writer=mywriter,dpi=200,bitrate=200)#,fps=2)#,dpi=100,bitrate=50)
					pl.close()

			if compare_2_run_movies:
				# create comparison between two sessions
				combinations = []
				for r1 in [self.runList[i] for i in self.conditionDict[condition]]:
					for r2 in [self.runList[i] for i in self.conditionDict[condition]]:
						if (r1.ID!=r2.ID):

							this_combo = [r1.ID,r2.ID]
							this_reverse_combo = [r2.ID,r1.ID]

							if this_combo not in combinations:

								combo_nifti = np.vstack([eval('nifti_session_%s'%r1.ID),eval('nifti_session_%s'%r2.ID)])
								self.logger.info('creating %s animation comparing run %s with run %s'%(condition,r1.ID,r2.ID))
								ims = []
								slice_dim = np.argmin(np.shape(combo_nifti))
								f=pl.figure(figsize=(4,4))
								timepoints = np.arange(len(combo_nifti))
								np.random.shuffle(timepoints)
								for t in timepoints[:500]:
									s=f.add_subplot(111)
									
									if slice_dim == 1:
										im=pl.imshow(combo_nifti[t,15,:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
									elif slice_dim == 2:
										im=pl.imshow(combo_nifti[t,:,15,:],origin='lowerleft',interpolation='nearest',cmap='gray')
									elif slice_dim == 3:
										im=pl.imshow(combo_nifti[t,:,:,15],origin='lowerleft',interpolation='nearest',cmap='gray')
									pl.axis('off')
									ims.append([im])
								ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
								mywriter = animation.FFMpegWriter(fps = 10)
								self.logger.info('saving to %s_%s_run_%s_vs_run_%s_registration_check.mp4'%('_'.join(postFix),condition,r1.ID,r2.ID))
								ani.save(os.path.join(plotdir,'%s_%s_run_%s_vs_run_%s_registration_check.mp4'%('_'.join(postFix),condition,r1.ID,r2.ID)),writer=mywriter,dpi=200,bitrate=200)#,fps=2)#,dpi=100,bitrate=50)
								pl.close()
								combinations.append(this_combo)
								combinations.append(this_reverse_combo)

			if compare_all_runs_movie:
				# combine all sessions into one animation
				all_epis = np.vstack(all_epis)
				t_order = np.arange(len(all_epis))
				np.random.shuffle(t_order)
				all_epis_shuffled = all_epis[t_order]

				self.logger.info('creating %s animation'%condition)
				pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
				ims = []
				f=pl.figure(figsize=(4,4))
				timepoints = np.arange(len(all_epis_shuffled))
				slice_dim = np.argmin(np.shape(all_epis_shuffled))
				for t in timepoints[:500]:
					s=f.add_subplot(111)
					if slice_dim == 1:
						im=pl.imshow(all_epis_shuffled[t,15,:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
					elif slice_dim == 2:
						im=pl.imshow(all_epis_shuffled[t,:,15,:],origin='lowerleft',interpolation='nearest',cmap='gray')
					elif slice_dim == 3:
						im=pl.imshow(all_epis_shuffled[t,:,:,15],origin='lowerleft',interpolation='nearest',cmap='gray')
					pl.axis('off')
					ims.append([im])
				ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
				mywriter = animation.FFMpegWriter(fps = 10)
				self.logger.info('saving to %s_%s_registration_check_all_runs.mp4'%('_'.join(postFix),condition))
				ani.save(os.path.join(plotdir,'%s_%s_registration_check_all_runs.mp4'%('_'.join(postFix),condition)),writer=mywriter)#,dpi=600,bitrate=100)#,fps=2)#)
				pl.close()


#########################################################################################################
# FOR COMPLEX NUISSANCE GLM
#########################################################################################################

	def resample_bet_mask(self):

		bet_mask_fn = self.runFile( stage = 'processed/mri', run =  self.runList[self.scanTypeDict['inplane_anat'][-1]], postFix = ['NB_mask'])
		target_format_fn = self.runFile(stage = 'processed/mri/', run = self.runList[self.scanTypeDict['epi_bold'][-1]] )
		bet_mask_ofn =  os.path.join(self.stageFolder(stage='processed/mri/masks/anat'),'bet_mask.nii.gz')

		fO = FlirtOperator(inputObject = bet_mask_fn,  referenceFileName = target_format_fn)
		fO.configureApply(outputFileName = bet_mask_ofn ) 
		fO.execute()

		# this image is now interpolated, lets binarize it again
		bet_mask = NiftiImage(bet_mask_ofn)
		bet_mask_bool = np.zeros_like(bet_mask.data)
		bet_mask_bool[bet_mask.data>0] = 1
		bet_mask_bool_nifti = NiftiImage(bet_mask_bool)
		bet_mask_bool_nifti.header = bet_mask.header
		bet_mask_bool_nifti.save(bet_mask_ofn)
		
	def create_WM_GM_CSF_masks(self,target_session=-1):
	
		# first, let's segment the target T2 into white matter, gray matter and csf
		T2_fn = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB'])
		FastO = FASTOperator(inputObject = T2_fn)
		FastO.configure()
		FastO.execute()

		# then we'll resample the binary masks down to functional space
		for matter_postFix in ['seg_0','seg_1','seg_2']:
			inputObject = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB',matter_postFix])
			outputObject = self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB',matter_postFix,'resampled'])
			fmO = FSLMathsOperator(inputObject=inputObject)
			fmO.configure(outputFileName=outputObject, **{'-subsamp2offc': ''})
			fmO.execute()

			# and now that it is smoothed (as a result of interpolation), let's binarize it again 
			# a threshold of 0.5 means some spatial specificity will remain
			mask = NiftiImage(outputObject)
			mask_bool = np.zeros_like(mask.data)
			mask_bool[mask.data>0.75] = 1
			mask_bool_nifti = NiftiImage(mask_bool)
			mask_bool_nifti.header = mask.header
			mask_bool_nifti.save(outputObject)

def prediction_per_slice(design_matrix, n_pixel_elements,rtime , ssr,slice_no, results_frames, hrf_params,prf_params, data,hrf_type,slices):

	voxels_in_this_slice = (slices == slice_no)
	these_hrf_params = hrf_params[:,voxels_in_this_slice]
	these_prf_params = prf_params[:,voxels_in_this_slice]
	these_data = data[:,voxels_in_this_slice]

	this_slice_predictions = []
	this_slice_r_squareds = []
	for voxno in range(voxels_in_this_slice.sum()):

		params = these_prf_params[:,voxno]
		if hrf_type != 'canonical':
			hrf_params = these_hrf_params[:,voxno]
		else:
			hrf_params = [1,0,0]

		g = gpf(design_matrix = design_matrix, max_eccentricity = 1, n_pixel_elements = n_pixel_elements, rtime = rtime, ssr = ssr,slice_no=slice_no)

		center_model_prediction = g.hrf_model_prediction(params[results_frames['xo']], params[results_frames['yo']], params[results_frames['sigma_center']],hrf_params)[0] * params[results_frames['amp_center']]
		surround_model_prediction = g.hrf_model_prediction(params[results_frames['xo']], params[results_frames['yo']], params[results_frames['sigma_surround']],hrf_params)[0] * params[results_frames['amp_surround']]
		combined_model_prediction = center_model_prediction + surround_model_prediction

		# see how well the prediction matches the data
		RSS =  np.sum((these_data[:,voxno] - combined_model_prediction)**2)
		r_squared = 1 - RSS/np.sum((these_data[:,voxno] - np.mean(these_data[:,voxno])) ** 2) 

		if (r_squared < 0) + np.isnan(r_squared):
			r_squared = 0.0

		this_slice_r_squareds.append(r_squared)
		# this_slice_predictions.append(combined_model_prediction)

	return this_slice_r_squareds#, this_slice_predictions


	def stimulus_prediction_per_run(self, postFix, n_pixel_elements = 101, sample_duration = 0.01, n_slices = 30,mask='V1',model='OG',voxel_specific_hrf=True):
		"""

		"""
		self.stimulus_timings() # this gives all the run objects their r.orientations / r.trial times etc
		mask_file = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask +  '.nii.gz'))
		cortex_mask = np.array(mask_file.data, dtype = bool)
		
		if voxel_specific_hrf:
			hrf_nifti_filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'hrf_parameters.nii.gz') 
			res_postFix = ['voxel_specific_hrf']
		else:
			hrf_nifti_filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'mean_hrf_parameters.nii.gz') 
			res_postFix = ['mean_hrf']
		all_hrf_parameters = NiftiImage(hrf_nifti_filename).data[:,cortex_mask]

		for ri,r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]): 

			this_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			# get run information, but cutoff first and last trial to make sure that sgtf artefacts are excluded
			TR = this_nii_file.rtime
			n_TRs = this_nii_file.timepoints

			if TR > 10:
				TR /= 1000
			elif TR < 0.01:
				TR *= 1000		

			# only take those trial times where there was a stimulus
			which_trials = (r.trial_times[:,0]!=3)
			r.trial_times = r.trial_times[which_trials]
			r.orientations = np.radians(r.orientations[which_trials])

			# first, let's create a design matrix for this run at TR resolution (will be upsampled later)
			bar_width = 0.25
			mr = PRFModelRun(r, n_TRs = n_TRs, TR = TR, n_pixel_elements = n_pixel_elements, sample_duration = TR, bar_width = bar_width)
			self.logger.info('simulating model experiment run %d with %d pixel elements and %1.2f s sample_duration'%(ri,n_pixel_elements, TR))
			mr.simulate_run()

			# now that we have a raw design matrix, we can start loading the prf parameters and generate per run predictions
			filename = 'results_'+ mask + '_' + '_'.join(postFix) + '_' + model + '_ALL'
			all_params = NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), filename +  '.nii.gz')).data[:,cortex_mask]
		
			slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
			ssr = np.round(1/(TR/float(n_slices)))

			all_predictions = []
			all_predictions_cortex = np.zeros([n_TRs] + list(cortex_mask.shape))

			import time as t
			start_time = t.time()
			percentage_done = 0
			for voxno in range(cortex_mask.sum()):

				params = all_params[:,voxno]
				hrf_params = all_hrf_parameters[:,voxno]
				g = gpf(design_matrix = mr.run_matrix, max_eccentricity = 1, n_pixel_elements = n_pixel_elements, rtime = TR, ssr = ssr,slice_no=slices[voxno])

				center_model_prediction = g.hrf_model_prediction(params[self.results_frames['xo']], params[self.results_frames['yo']], params[self.results_frames['sigma_center']],hrf_params)[0] * params[self.results_frames['amp_center']]
				surround_model_prediction = g.hrf_model_prediction(params[self.results_frames['xo']], params[self.results_frames['yo']], params[self.results_frames['sigma_surround']],hrf_params)[0] * params[self.results_frames['amp_surround']]
				combined_model_prediction =  center_model_prediction + surround_model_prediction 

				all_predictions.append(combined_model_prediction)

				if percentage_done < ceil(100 * voxno / cortex_mask.sum()):
					percentage_done = percentage_done + 1
					estimated_time_left = (((t.time()-start_time) / percentage_done) * (100-percentage_done))/60.
					self.logger.info( 'voxel # ' + str(voxno) + ' done ' + str(percentage_done) + ' percent, eta: %.2fmin'%estimated_time_left )
			

			self.logger.info('done fitting in %.2f min'%((t.time() - start_time)/60.) )
			self.logger.info('saving predictions' )

			all_predictions_cortex[:,cortex_mask] = np.array(all_predictions).T

			this_run_prediction_nifti = NiftiImage(all_predictions_cortex)
			this_run_prediction_nifti.header = this_nii_file.header
			this_run_prediction_nifti.save(self.runFile(stage = 'processed/mri', run = r, postFix = ['predictions'] + res_postFix ))

	def compare_eye_regressors(self):

		V1 = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'V1.nii.gz')).data.astype(bool)
		r = self.runList[0]
		data = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc'] )).data[:,V1]
		nn = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yyy'] )).data[:,V1]
		yn = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yn'] )).data[:,V1]
		ny = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_ny'] )).data[:,V1]
		yy = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yy'] )).data[:,V1]

		RSS_nn = np.sum(nn**2)
		RSS_yn = np.sum(yn**2)
		RSS_ny = np.sum(ny**2)
		RSS_yy = np.sum(yy**2)

		R2_nn = np.sum((data - nn)**2)
		R2_yn = np.sum((data - yn)**2)
		R2_ny = np.sum((data - ny)**2)
		R2_yy = np.sum((data - yy)**2)

		nn_k = 19
		yn_k = ny_k = 20
		yy_k = 21

		n = np.sum(V1)*765

		AIC_nn = n*np.log(RSS_nn/n) + 2*nn_k
		AIC_yn = n*np.log(RSS_yn/n) + 2*yn_k
		AIC_ny = n*np.log(RSS_ny/n) + 2*ny_k
		AIC_yy = n*np.log(RSS_yy/n) + 2*yy_k

		BIC_nn = n*np.log(RSS_nn/n) + 2*nn_k * np.log(n)
		BIC_yn = n*np.log(RSS_yn/n) + 2*yn_k * np.log(n)
		BIC_ny = n*np.log(RSS_ny/n) + 2*ny_k * np.log(n)
		BIC_yy = n*np.log(RSS_yy/n) + 2*yy_k * np.log(n)

		f=pl.figure(figsize = (20,10))
		s = f.add_subplot(411)
		pl.bar(np.arange(4),[AIC_nn,AIC_yn,AIC_ny,AIC_yy])
		pl.xticks(np.arange(4)+0.5,['AIC_nn','AIC_yn','AIC_ny','AIC_yy'])
		pl.title('AICs of residuals')
		pl.ylabel('AIC')
		pl.ylim(0.9*np.min([AIC_nn,AIC_yn,AIC_ny,AIC_yy]),1.1*np.max([AIC_nn,AIC_yn,AIC_ny,AIC_yy]))

		s = f.add_subplot(412)
		pl.bar(np.arange(4),[BIC_nn,BIC_yn,BIC_ny,BIC_yy])
		pl.xticks(np.arange(4)+0.5,['BIC_nn','BIC_yn','BIC_ny','BIC_yy'])
		pl.title('BICs of residuals')
		pl.ylabel('BIC')
		pl.ylim(0.9*np.min([BIC_nn,BIC_yn,BIC_ny,BIC_yy]),1.1*np.max([BIC_nn,BIC_yn,BIC_ny,BIC_yy]))

		s = f.add_subplot(413)
		pl.bar(np.arange(4),[RSS_nn,RSS_yn,RSS_ny,RSS_yy])
		pl.xticks(np.arange(4)+0.5,['RSS_nn','RSS_yn','RSS_ny','RSS_yy'])
		pl.title('sum of residuals of GLMs')
		pl.ylabel('RSS')
		pl.ylim(0.9*np.min([RSS_nn,RSS_yn,RSS_ny,RSS_yy]),1.1*np.max([RSS_nn,RSS_yn,RSS_ny,RSS_yy]))

		s = f.add_subplot(414)
		pl.bar(np.arange(4),[R2_nn,R2_yn,R2_ny,R2_yy])
		pl.xticks(np.arange(4)+0.5,['R2_nn','R2_yn','R2_ny','R2_yy'])
		pl.title('explained variance by GLM')
		pl.ylabel('R2')
		pl.ylim(0.9*np.min([R2_nn,R2_yn,R2_ny,R2_yy]),1.1*np.max([R2_nn,R2_yn,R2_ny,R2_yy]))

		pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'), 'eye_GLM_comparison_AIC.pdf'))


		pl.figure(figsize=(20,10))
		pl.plot(nn[:,19],linewidth=0.8)
		pl.plot(yn[:,19],linewidth=0.8)
		pl.plot(yy[:,19],linewidth=0.8)
		pl.legend(['nn','yn','yy'],loc='best')

		pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'), 'nuisance_GLM_eye_eval.pdf'))

	def compare_dt_ddt_moco(self):

		V1 = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'V1.nii.gz')).data.astype(bool)
		r = self.runList[0]
		data = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc'] )).data[:,V1]
		nnn = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_nnn'] )).data[:,V1]
		ynn = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_ynn'] )).data[:,V1]
		nyn =  NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_nyn'] )).data[:,V1]
		nny = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_nny'] )).data[:,V1]
		yyn =  NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yyn'] )).data[:,V1]
		yny =  NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yny'] )).data[:,V1]
		nyy =  NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_nyy'] )).data[:,V1]
		yyy =  NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','sgtf','psc','betas_yyy'] )).data[:,V1]

		RSS_nnn = np.sum(nnn**2)
		RSS_ynn = np.sum(ynn**2)
		RSS_nyn = np.sum(nyn**2)
		RSS_nny = np.sum(nny**2)
		RSS_yyn = np.sum(yyn**2)
		RSS_nyy = np.sum(nyy**2)
		RSS_yny = np.sum(yny**2)
		RSS_yyy = np.sum(yyy**2)

		R2_nnn = np.sum((data - nnn)**2)
		R2_ynn = np.sum((data - ynn)**2)
		R2_nyn = np.sum((data - nyn)**2)
		R2_nny = np.sum((data - nny)**2)
		R2_yyn = np.sum((data - yyn)**2)
		R2_nyy = np.sum((data - nyy)**2)
		R2_yny = np.sum((data - yny)**2)
		R2_yyy = np.sum((data - yyy)**2)

		nnn_k = 1
		ynn_k = nyn_k = nny_k = 7
		yyn_k = yny_k = nyy_k = 13
		yyy_k = 19

		n = np.sum(V1)*765

		AIC_nnn = n*np.log(RSS_nnn/n) + 2*nnn_k
		AIC_ynn = n*np.log(RSS_ynn/n) + 2*ynn_k
		AIC_nyn = n*np.log(RSS_nyn/n) + 2*nyn_k
		AIC_nny = n*np.log(RSS_nny/n) + 2*nny_k
		AIC_yyn = n*np.log(RSS_yyn/n) + 2*yyn_k
		AIC_nyy = n*np.log(RSS_nyy/n) + 2*nyy_k
		AIC_yny = n*np.log(RSS_yny/n) + 2*yny_k
		AIC_yyy = n*np.log(RSS_yyy/n) + 2*yyy_k

		BIC_nnn = n*np.log(RSS_nnn/n) + 2*nnn_k * np.log(n)
		BIC_ynn = n*np.log(RSS_ynn/n) + 2*ynn_k * np.log(n)
		BIC_nyn = n*np.log(RSS_nyn/n) + 2*nyn_k * np.log(n)
		BIC_nny = n*np.log(RSS_nny/n) + 2*nny_k * np.log(n)
		BIC_yyn = n*np.log(RSS_yyn/n) + 2*yyn_k * np.log(n)
		BIC_nyy = n*np.log(RSS_nyy/n) + 2*nyy_k * np.log(n)
		BIC_yny = n*np.log(RSS_yny/n) + 2*yny_k * np.log(n)
		BIC_yyy = n*np.log(RSS_yyy/n) + 2*yyy_k * np.log(n)

		f=pl.figure(figsize = (20,10))
		s = f.add_subplot(411)
		pl.bar(np.arange(8),[AIC_nnn,AIC_ynn,AIC_nyn,AIC_nny,AIC_yyn,AIC_yny,AIC_nyy,AIC_yyy])
		pl.xticks(np.arange(8)+0.5,['AIC_nnn','AIC_ynn','AIC_nyn','AIC_nny','AIC_yyn','AIC_yny','AIC_nyy','AIC_yyy'])
		pl.title('AICs of residuals')
		pl.ylabel('AIC')
		pl.ylim(0.9*np.min([AIC_nnn,AIC_ynn,AIC_nyn,AIC_nny,AIC_yyn,AIC_yny,AIC_nyy,AIC_yyy]),1.1*np.max([AIC_nnn,AIC_ynn,AIC_nyn,AIC_nny,AIC_yyn,AIC_yny,AIC_nyy,AIC_yyy]))

		s = f.add_subplot(412)
		pl.bar(np.arange(8),[BIC_nnn,BIC_ynn,BIC_nyn,BIC_nny,BIC_yyn,BIC_yny,BIC_nyy,BIC_yyy])
		pl.xticks(np.arange(8)+0.5,['BIC_nnn','BIC_ynn','BIC_nyn','BIC_nny','BIC_yyn','BIC_yny','BIC_nyy','BIC_yyy'])
		pl.title('BICs of residuals')
		pl.ylabel('BIC')
		pl.ylim(0.9*np.min([BIC_nnn,BIC_ynn,BIC_nyn,BIC_nny,BIC_yyn,BIC_yny,BIC_nyy,BIC_yyy]),1.1*np.max([BIC_nnn,BIC_ynn,BIC_nyn,BIC_nny,BIC_yyn,BIC_yny,BIC_nyy,BIC_yyy]))

		s = f.add_subplot(413)
		pl.bar(np.arange(8),[RSS_nnn,RSS_ynn,RSS_nyn,RSS_nny,RSS_yyn,RSS_yny,RSS_nyy,RSS_yyy])
		pl.xticks(np.arange(8)+0.5,['RSS_nnn','RSS_ynn','RSS_nyn','RSS_nny','RSS_yyn','RSS_yny','RSS_nyy','RSS_yyy'])
		pl.title('sum of residuals of GLMs')
		pl.ylabel('RSS')
		pl.ylim(0.9*np.min([RSS_nnn,RSS_ynn,RSS_nyn,RSS_nny,RSS_yyn,RSS_yny,RSS_nyy,RSS_yyy]),1.1*np.max([RSS_nnn,RSS_ynn,RSS_nyn,RSS_nny,RSS_yyn,RSS_yny,RSS_nyy,RSS_yyy]))

		s = f.add_subplot(414)
		pl.bar(np.arange(8),[R2_nnn,R2_ynn,R2_nyn,R2_nny,R2_yyn,R2_yny,R2_nyy,R2_yyy])
		pl.xticks(np.arange(8)+0.5,['R2_nnn','R2_ynn','R2_nyn','R2_nny','R2_yyn','R2_yny','R2_nyy','R2_yyy'])
		pl.title('explained variance by GLM')
		pl.ylabel('R2')
		pl.ylim(0.9*np.min([R2_nnn,R2_ynn,R2_nyn,R2_nny,R2_yyn,R2_yny,R2_nyy,R2_yyy]),1.1*np.max([R2_nnn,R2_ynn,R2_nyn,R2_nny,R2_yyn,R2_yny,R2_nyy,R2_yyy]))

		pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'), 'moco_GLM_comparison_AIC.pdf'))



#########################################################################################################
# REST
#########################################################################################################

	def inflate_T2s(self,):
		
		for T2_run in self.scanTypeDict['inplane_anat']:
			
			if self.runList[T2_run].ID != 11:
				T2_fn = self.runFile( stage = 'processed/mri', run =  self.runList[T2_run], postFix = ['to_11_NB'])
				T2_ofn = self.runFile( stage = 'processed/mri', run =  self.runList[T2_run], postFix = ['to_11_NB_inflated'])
			else:
				T2_fn = self.runFile( stage = 'processed/mri', run =  self.runList[T2_run], postFix = ['NB'])
				T2_ofn = self.runFile( stage = 'processed/mri', run =  self.runList[T2_run], postFix = ['NB_inflated'])

			vsO = VolToSurfOperator(inputObject = T2_fn)
			vsO.configure(outputFileName = T2_ofn, threshold = 0.0, surfSmoothingFWHM = 0,frames = {'_f':0}, surfType = 'paint' ,register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ) )
			vsO.execute()

	def check_t_pulses(self):
		
		"""
		check t pulses looks for t pulses in the pickle behavior file and plots them against the stimulus timings.
		This way we can check whether the t_pulse transmitter failed at some point, which is important as it could 
		potentially mess with stimulus timings if t-pulses at the start of the experiment failed.		
		"""

		for condition in self.run_types:
			run_start_time = []
			exp_end_time = []
			fig = pl.figure(figsize=(12,6))	
			f2 = pl.figure(figsize=(12,8))
			for ri, r in enumerate([self.runList[i] for i in self.conditionDict[condition]]):
				filename = self.runFile(stage = 'processed/behavior', run = r, extension = '.dat' )
				with open(filename) as f:
					picklefile = pickle.load(f)
				run_start_time_string = [e for e in picklefile['eventArray'][0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
				run_start_time.append(float(run_start_time_string[0].split(' ')[-1]))

				# check whether pulses came through
				t_pulse=[]
				for ti in range(size(picklefile['eventArray'])):
					if 'Square' in self.project.projectName:
						t_pulse.append([float(e.split(' ')[-1]) - run_start_time[ri] for e in picklefile['eventArray'][ti] if "'unicode': u't'" in e])
					else:
						t_pulse.append([float(e.split(' ')[-1]) - run_start_time[ri] for e in picklefile['eventArray'][ti] if "event t at" in e])
				
				if 'Square' in self.project.projectName:
					end_exp_string = [e for e in picklefile['eventArray'][39] if e[:len('trial 39 phase 2')] == 'trial 39 phase 2']
					exp_end_time.append(float(end_exp_string[0].split(' ')[-1])- run_start_time[ri]+43.8)
				else:
					end_exp_string = [e for e in picklefile['eventArray'][47] if e[:len('trial 47 phase 3')] == 'trial 47 phase 3']
					exp_end_time.append(float(end_exp_string[0].split(' ')[-1])- run_start_time[ri]+3.8)

				t_pulse = np.hstack(t_pulse)

				trial_starts = []
				trial_start_string = [[e for e in picklefile['eventArray'][i] if 'phase 1' in e ] for i in range(len(picklefile['eventArray'])) ]
				trial_start_string = [[e for e in picklefile['eventArray'][i] if 'phase 1' in e ] for i in range(len(picklefile['eventArray'])) ]
				trial_starts = [ float(ts[0].split(' ')[-1])-run_start_time[ri] for ts in trial_start_string ]
				# trial_starts.append(float(trial_start_string[0][0].split(' ')[-1]))

				rounded_start_times = (np.round(trial_starts,1)*10).astype(int)
				start_times_array = np.zeros(np.max(rounded_start_times)+1)
				start_times_array[rounded_start_times]=1

				rounded_pulses = (np.round(t_pulse,1)*10).astype(int)
				t_pulse_array = np.zeros(np.max(rounded_pulses)+1)
				t_pulse_array[rounded_pulses]=1

				# print '%d t-pulses occured after the last trial start in run %d'%(len(t_pulse[t_pulse>trial_starts[-1]]),ri)
				
				s1 = fig.add_subplot(len(self.conditionDict[condition]),1,ri)
				s1.plot(t_pulse_array,'b')
				s1.plot(start_times_array,'r')
				s1.set_ylim(0,2)

				s2 = f2.add_subplot(len(self.conditionDict[condition]),1,ri)
				num_ts = t_pulse.size
				niftiImage =  NiftiImage(self.runFile(stage = 'processed/mri', run = r))
				num_trs = niftiImage.getTimepoints()
				if num_ts != num_trs:
					print '!!!! ERROR !!!!! \n In run %d, num_trs (%d) != num_ts (%d)'%(ri,num_trs,num_ts)
				else:
					print 'Run %d is fine'%ri
				# print 'exp ended at %d, last t-pulse was at %d'%(exp_end_time[ri],t_pulse[-1]) 

				# s = fig.add_subplot(len(self.conditionDict[condition]),1,ri)
				s2.hist(np.diff(t_pulse),color='#c94545')
				# pl.xlim(0,15)
				simpleaxis(s2)
				spine_shift(s2)
			fig.savefig(os.path.join(self.stageFolder('processed/mri/figs/'),'check_t_pulses_%s.pdf'%condition))
			f2.savefig(os.path.join(self.stageFolder('processed/mri/figs/'),'check_t_pulses_2_%s.pdf'%condition))

	
	def slice_time_correct(self,condition='Mapper',postFix=['mcf','warped','sgtf','psc']):

		stcOs = []
		for r in [self.runList[i] for i in self.conditionDict[condition]]: 

			inputfilename = self.runFile(stage = 'processed/mri', run = r, postFix = postFix)
			outputfilename = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['stc'])
			TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).rtime
			if TR > 1000:
				TR /= 1000.
			if TR < 0.01:
				TR *= 1000

			stcO = SliceTimeCorrectionOperator(inputObject = inputfilename)
			stcO.configure(outputFileName = outputfilename, TR = TR)
			stcOs.append(stcO)		
				
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers, secret='mc')
		self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
		ppResults = [job_server.submit(ExecCommandLine,(stcO.runcmd,),(),('subprocess','tempfile',)) for stcO in stcOs]
		for fMcf in ppResults:
			fMcf()
		job_server.print_stats()








#########################################################################################################
# HRF ESTIMATION
#########################################################################################################

	def fit_hrf_params(self,mask,postFix,model,ssr=20,add_empty_trs=20):

		orientations = ['0','45','90','135','180','225','270','315','X']
		n_orientations = len(orientations)

		# load stats
		self.logger.info('loading statistical mask')
		filename =os.path.join(self.stageFolder('processed/mri/PRF/'), 'corrs_' + mask + '_' + '_'.join(postFix)  + '_' + model + '_ALL.nii.gz')
		corrs = NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), filename )).data[self.stats_frames['r_squared']]
		corr_threshold = np.sort(np.ravel(corrs))[-3001]
		cortex_mask = (corrs> corr_threshold)

		rmps = np.hstack(NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), 'raw_model_predictions_' + mask + '_' + '_'.join(postFix)  + '_' + model + '_ALL.nii.gz')).data[:,cortex_mask].T)
		data =  np.hstack(NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), 'averaged_data_' + mask + '_' + '_'.join(postFix) + '_ALL.nii.gz' )).data[:,cortex_mask].T)

		# add empty trs between trials, so that hrf dilation does not bother with the next trial
		tr_per_trial = len(rmps)/(n_orientations*cortex_mask.sum())
		padded_rmp = np.zeros(len(rmps)+add_empty_trs*n_orientations*cortex_mask.sum())
		padd_mask = np.zeros(len(padded_rmp)).astype(bool)
		for i in range(n_orientations*cortex_mask.sum()):
			padded_rmp[i*tr_per_trial+add_empty_trs*i:(i+1)*tr_per_trial+add_empty_trs*i] = rmps[i*tr_per_trial:(i+1)*tr_per_trial]
			padd_mask[i*tr_per_trial+add_empty_trs*i:(i+1)*tr_per_trial+add_empty_trs*i] = True		

		def residual(params,data,rmps,padd_mask):
			
			rmp = np.repeat(rmps, ssr, axis=0)
			hrf_kernel = doubleGamma(np.arange(0,32,self.TR/float(ssr)),
				params['hrf_a1'].value,params['hrf_a2'].value,params['hrf_b1'].value,params['hrf_b2'].value,params['hrf_c'].value)
			if hrf_kernel.shape[0] % 2 == 1:
				hrf_kernel = np.r_[hrf_kernel, 0]
			hrf_kernel /= np.abs(hrf_kernel).sum()

			convolved_mp = fftconvolve( rmp, hrf_kernel, 'full' )[:rmp.shape[0]:ssr]

			return data - convolved_mp[padd_mask]

		params = Parameters()
		params.add('hrf_a1',expr='hrf_d1/hrf_b1')
		params.add('hrf_a2',expr='hrf_d2/hrf_b2')
		params.add('hrf_b1',value=0.9)
		params.add('hrf_b2',value=0.9)
		params.add('hrf_c',value=0.35,min=0)
		params.add('hrf_d1',value=5.4)
		params.add('hrf_d2',value=10.8,expr='hrf_d1+delta_hrf_d')
		params.add('delta_hrf_d',value=5.4,min=0)

		self.logger.info('now fitting hrf parameters on data from %d voxels'%cortex_mask.sum())
		import time 
		t1 = time.time()
		minimize(residual, params, args=(), kws={'data':data,'rmps':padded_rmp,'padd_mask':padd_mask},method='powell')
		t2 = time.time()
		self.logger.info('fitted hrf parameters in %.2f minutes'%((t2-t1)/60.))

		param_dict = {}
		for param in params:
			param_dict[param] = params[param].value

		with open(os.path.join(self.stageFolder('processed/mri/PRF/'), 'hrf_params_%s_%s.pickle'%(mask,'_'.join(postFix))), 'w') as f:
			pickle.dump({'hrf_params':param_dict}, f)

		hrf_kernel = doubleGamma(np.arange(0,32,self.TR/float(ssr)),
			params['hrf_a1'].value,params['hrf_a2'].value,params['hrf_b1'].value,params['hrf_b2'].value,params['hrf_c'].value)
		canonical_hrf_kernel = doubleGamma(np.arange(0,32,self.TR/float(ssr)))
		plotdir = self.stageFolder('processed/mri/figs')
		f = pl.figure(figsize=(12,6))
		s = f.add_subplot(111)
		pl.plot(hrf_kernel,label='subject specific HRF')
		pl.plot(canonical_hrf_kernel,'--k',label='canonical HRF')
		s.legend(fancybox = True, loc = 'best')
		pl.xticks(np.linspace(0,len(hrf_kernel),16),np.arange(0,32,2))
		pl.xlabel('time (s)')
		# pl.axis('off')
		s.text(len(hrf_kernel)*0.6,0.8,'\nHRF parameters: \n\na1: %.2f\na2: %.2f\nb1: %.2f\nb2: %.2f\nc: %.2f'
			 %(params['hrf_a1'],params['hrf_a2'],params['hrf_b1'],params['hrf_b2'],params['hrf_c']),horizontalalignment='center',verticalalignment='center',fontsize=12,bbox={'facecolor':'white', 'alpha':1, 'pad':10})
		pl.savefig(os.path.join(plotdir,'HRF'))

def hrf_fit_one_voxel(params,data,n_slices,total_TRs,TR,regressor_list,sl,baselines):

	def residual(params,data,n_slices,total_TRs,TR,regressor_list,sl,baselines):
		

		## create design matrix
		run_design = NewDesign(total_TRs, TR, sample_duration = TR/float(n_slices))
		run_design.configure(regressor_list, hrf_parameters = [params['hrf'].value,params['hrf_dt'].value,params['hrf_ddt'].value])

		joined_design_matrix = np.vstack([baselines,run_design.convolved_design_matrix[:,sl:total_TRs*n_slices:n_slices]]).T

		dm = np.mat(joined_design_matrix)

		betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(data).T
		residuals = data - (np.mat(dm) * np.mat(betas))

		return residuals

	params = Parameters()
	params.add('hrf',value=1)
	params.add('hrf_dt',value=0)
	params.add('hrf_ddt',value=0)

	minimize(residual, params, args=(), kws={'data':data,'n_slices':n_slices,'total_TRs':total_TRs,'TR':TR,'regressor_list':regressor_list,'sl':sl,'baselines':baselines},method='powell')

	return params['hrf'].value, params['hrf_dt'].value, params['hrf_ddt'].value


	def compare_hrf_methods(self,mask='combined_labels',postFix=['mcf','warped','sgtf','psc'],model='OG'):

		mean_hrf_corrs = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'corrs_' + mask + '_' + '_'.join(postFix)  + '_' + model + '_ALL.nii.gz')).data[0]
		voxel_hrf_corrs = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'corrs_' + mask + '_' + '_'.join(postFix) + '_voxel_specific_hrf'  + '_' + model + '_ALL.nii.gz')).data[0]
		
		f = pl.figure(figsize=(20,12))
		for mi,mask in enumerate(['V1','V2','V3','V4','LO','MT','V3AB','VO','PHC','IPS0','IPS1','IPS2','IPS3','IPS4']):#,'IPS5','FEF']
		
			s = f.add_subplot(3,5,mi+1)
			cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder( stage = 'processed/mri/masks/anat'), mask)).data, dtype = bool)

			pl.title(mask)
			pl.plot(mean_hrf_corrs[cortex_mask],voxel_hrf_corrs[cortex_mask],'o')
			pl.plot([0,1],[0,1],'--k')
			pl.xlim(0,1)
			pl.ylim(0,1)
			pl.xlabel('median hrf')
			pl.ylabel('voxel specific hrf')

		pl.tight_layout()

		pl.savefig(os.path.join(self.stageFolder( stage = 'processed/mri/figs/'),'compare_hrf_methods.pdf'))


# TAKEN FROM HRF_FROM_MAPPER FUNCTION:
##### one hrf approach:
if method == 'one_hrf':

	self.logger.info('now fitting one HRF on data from %d voxels'%cortex_mask.sum())


	def residual(params,data,n_slices,total_TRs,TR,regressor_list,cortex_mask,baselines):
		

		residuals = np.zeros_like(all_data)
		## create design matrix
		run_design = NewDesign(total_TRs, TR, sample_duration = TR/float(n_slices))
		run_design.configure(regressor_list, hrf_parameters = [params['hrf'].value,params['hrf_dt'].value,params['hrf_ddt'].value])

		for sl in range(n_slices):
			slice_mask = np.zeros_like(cortex_mask)
			slice_mask[sl,:,:] = cortex_mask[sl,:,:]
			
			if slice_mask.sum() > 0:

				joined_design_matrix = np.vstack([baselines,run_design.convolved_design_matrix[:,sl:total_TRs*n_slices:n_slices]]).T

				dm = np.mat(joined_design_matrix)

				betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(all_data[:,slice_mask].T).T
				residuals[:,slice_mask] = all_data[:,slice_mask] - (np.mat(dm) * np.mat(betas))

		return np.ravel(residuals)

	import time 
	t1 = time.time()
	minimize(residual, params, args=(), kws={'data':all_data,'n_slices':n_slices,'total_TRs':total_TRs,'TR':TR,'regressor_list':regressor_list,'cortex_mask':cortex_mask,'baselines':baselines},method='powell')
	t2 = time.time()
	self.logger.info('fitted hrf parameters in %.2f minutes'%((t2-t1)/60.))

	xx = np.arange(0,32,0.05)
	fitted_hrf = params['hrf'].value * he.hrf.spmt(xx)[:, None] + params['hrf_dt'].value  * he.hrf.dspmt(xx)[:, None] + params['hrf_ddt'].value  * he.hrf.ddspmt(xx)[:, None]
	fitted_hrf /= np.sum(np.abs(fitted_hrf))
	standard_SPM_hrf = he.hrf.spmt(xx)[:, None]
	standard_SPM_hrf /= np.sum(np.abs(standard_SPM_hrf))
	doublegamma_hrf = doubleGamma(xx)
	doublegamma_hrf /= np.sum(np.abs((doublegamma_hrf)))	
	f = pl.figure(figsize=(8,8))
	s = f.add_subplot(111)	
	pl.plot(xx,fitted_hrf,label='fitted hrf')
	pl.plot(xx,doublegamma_hrf,linestyle='--',label='doublegamma')
	pl.plot(xx,standard_SPM_hrf,linestyle='--',label='SPM standard')
	pl.legend(loc='best')
	pl.xlabel('time (s)')
	pl.ylabel('a.u.')
	pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'),'fitted_HRF.pdf'))

	hrf_params = [params['hrf'].value,params['hrf_dt'].value,params['hrf_ddt'].value]
	all_mean_hrfs = np.reshape(np.tile(hrf_params,(np.size(cortex_mask),1)).T,([3] + list(cortex_mask.shape)))
	
	self.logger.info('outputting mean hrf parameters')			
	mean_hrf_nii_file = NiftiImage(all_mean_hrfs)
	mean_hrf_nii_file.header = header
	mean_hrf_nii_file.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'mean_hrf_parameters.nii.gz'))

########## 1 hrf per voxel (parallel) TAKES AGES!
elif method == 'per_voxel_hrf':

	self.logger.info('now fitting one HRF per voxel')

	res = Parallel(n_jobs = n_jobs, verbose = 9)(
				delayed(hrf_fit_one_voxel)
				(params=params,
				data=all_data[:,cortex_mask][:,voxno],
				n_slices=n_slices,
				total_TRs=total_TRs,
				TR=TR,
				regressor_list=regressor_list,
				sl=int((np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask][voxno]),
				baselines=baselines)
					for voxno in np.arange(cortex_mask.sum()))



	def feat_analysis_mapper(self,feat_file='all_mapper1.fsf',postFix = ['mcf','sgtf','res']):

		for r in [self.runList[i] for i in self.conditionDict['Mapper']]: 
			try:
				self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.fsf'))
			except OSError:
				pass

			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			thisFeatFile = 'fsfs/' + feat_file
		
			TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).rtime
			if TR > 1000:
				TR /= 1000.
			if TR < 0.01:
				TR *= 1000

			REDict = {
			'---nii_file---': 						self.runFile(stage = 'processed/mri', run = r, postFix = postFix), 
			'---TR---':								str(TR),
			'---n_TRs---':							str(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints),
			'---fix_no_stim_file---': 					os.path.join(self.runFolder(stage = 'processed/mri',run=r),'fix_no_stim.txt'), 	
			'---no_color_no_speed_file---': 		os.path.join(self.runFolder(stage = 'processed/mri',run=r),'no_color_no_speed.txt'), 	
			'---no_color_yes_speed_file---': 		os.path.join(self.runFolder(stage = 'processed/mri',run=r),'no_color_yes_speed.txt'), 	
			'---yes_color_no_speed_file---': 		os.path.join(self.runFolder(stage = 'processed/mri',run=r),'yes_color_no_speed.txt'), 
			'---yes_color_yes_speed_file---':		os.path.join(self.runFolder(stage = 'processed/mri',run=r),'yes_color_yes_speed.txt'), 	
			}
			featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
			featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
			self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
			# run feat
			featOp.execute()

	def apply_registration_for_feat(self,postFix):

		for r in [self.runList[i] for i in self.conditionDict['Mapper']]:

			feat_dir = featFileName = self.runFile(stage = 'processed/mri', run = r, extension = '.feat', postFix = postFix)
			self.setupRegistrationForFeat(feat_dir)

	def warp_back_to_subject_space_and_project_to_surf(self,stat_file_names = ['cope', 'tstat','zstat','thresh_zstat'],num_copes=11):#['cope', 'tstat','zstat','cluster_mask_zstat']

		output_base_dir = self.stageFolder(stage='processed/mri/Mapper/feat_results/')
		if not os.path.isdir(output_base_dir): os.mkdir(output_base_dir)
		
		feat_base_dir = self.stageFolder(stage='processed/mri/Mapper/all.gfeat')
		copes=['cope%d.feat'%(ci+1) for ci in range(num_copes)]
		for ci,cope in enumerate(copes):
			output_dir = os.path.join(output_base_dir,cope.split('.')[0])
			if os.path.isdir(output_dir): shutil.rmtree(output_dir)
			os.mkdir(output_dir)
			for stat_name in stat_file_names:
				# apply the transform
				if not 'thresh' in stat_name:
					inputfn = os.path.join(feat_base_dir,cope,'stats', stat_name + str(1)+'.nii.gz')
				else:
					inputfn = os.path.join(feat_base_dir,cope, stat_name + str(1) +'.nii.gz')

				outputfn = os.path.join(output_dir, stat_name+str(ci+1)+'.nii.gz')
				targetfn = self.runFile(stage = 'processed/mri/', run = self.runList[self.conditionDict['Mapper'][0]], postFix = ['mcf','warped'])
				flO = FlirtOperator(inputObject = inputfn, referenceFileName =targetfn)
				flO.configureApply(self.runFile(stage = 'processed/mri/reg/feat', base = 'standard2example_func', extension = '.mat' ), 
										outputFileName = outputfn )
				flO.execute()

				if not 'thresh' in stat_name:

					for sm in [0,3,5]:
						# and project to surface
						vsO = VolToSurfOperator(inputObject = os.path.join(output_dir, stat_name+str(ci+1)+'.nii.gz'))
						ofn = os.path.join(os.path.join(output_dir, stat_name+str(ci+1)) + '_sm_' + str(sm))
						vsO.configure(outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = sm,frames = {'_f':0}, surfType = 'paint' ,register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ) )
						vsO.execute()

				else:

					# this image is now interpolated, lets binarize it again
					mask = NiftiImage(outputfn)
					mask_bool = np.zeros_like(mask.data)
					mask_bool[mask.data>2] = 1
					mask_bool_nifti = NiftiImage(mask_bool)
					mask_bool_nifti.header = mask.header
					mask_bool_nifti.save(outputfn)


	def PRF_nuisance_GLM(self, mask = 'bet_mask',postFix = ['mcf','fnirted','sgtf'],target_session =-1,plot=True,voxel_specific_hrf=True,n_jobs=20,model='OG'):
		"""GLM_for_nuisances takes a diverse set of nuisance regressors,
		runs a GLM on them in order to run further PRF analysis on the 
		residuals after GLM. 

		At the moment, it incorporates the motion correction parameters 
		and eye blinks. Physio yet needs to be implemented.
		"""
		# self.stimulus_timings()
		# self.eye_informer()
		cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask + '.nii.gz')).data, dtype = bool)
		# filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'corrs_' + mask + '_' + '_'.join(base_on_postFix)  + '_' + model + '_ALL')
		# corrs = NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), filename +  '.nii.gz')).data[self.stats_frames['r_squared']]
		# filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'hrf_params.pickle'%(mask_file_name,'_'.join(postFix))) 
		# with open(filename) as f:
		# 	picklefile = pickle.load(f)
		# hrf_params = picklefile['hrf_params']


		# figure out roi label per voxel and how many there are for when plotting individual voxels in Dumoulin_fit
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		anatRoiFileNames = [anRF for anRF in anatRoiFileNames if np.all([np.any(['bh' in anRF,'lh' in anRF,'rh' in anRF]),'cortex' not in anRF])]
		roi_names = np.zeros_like(cortex_mask).astype('string')
		for this_roi in anatRoiFileNames:
			roi_nifti = NiftiImage(this_roi).data.astype('bool')
			roi_names[roi_nifti] = (this_roi.split('/')[-1]).split('.')[1]
		roi_names[roi_names=='False'] = 'unkown_roi'
		# roi_names = roi_names[cortex_mask]

		if voxel_specific_hrf:
			hrf_nifti_filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'hrf_parameters.nii.gz') 
			res_postFix = ['voxel_specific_hrf']
		else:
			hrf_nifti_filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'mean_hrf_parameters.nii.gz') 
			res_postFix = ['mean_hrf']
		all_hrf_parameters = NiftiImage(hrf_nifti_filename).data

		for r in [self.runList[i] for i in self.conditionDict['PRF']]:

			if plot:
				this_plot_dir = os.path.join(self.runFolder('processed/mri/',run=r),'nuisance_GLM_%s'%res_postFix[0])
				if os.path.isdir(this_plot_dir): shutil.rmtree(this_plot_dir)
				os.mkdir(this_plot_dir)
			else:
				this_plot_dir = []

			self.logger.info('loading %s'%self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			## load nifti data
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
			nii_data = nii_file.data
			if nii_file.rtime > 1000:
				rtime = nii_file.rtime/1000
			elif nii_file.rtime < 0.01:
				rtime = nii_file.rtime * 1000
			else:
				rtime = nii_file.rtime
			n_slices = nii_file.volextent[-1]
			n_timepoints = nii_file.timepoints

			## for PRF, load the predictions
			stim_predictions = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['predictions'] + res_postFix )).data

			## transients
			self.logger.info('loading transient times')
			transients_0 = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'transient_times_0.txt'))
			transients_0_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in transients_0]
			transients_1 = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'transient_times_1.txt'))
			transients_1_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in transients_1]
			transients_2 = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'transient_times_2.txt'))
			transients_2_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in transients_2]

			## motion correction parameters
			self.logger.info('loading motion correction parameters')
			mcf = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' ))
			mcf_dt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_dt.par' ))
			mcf_ddt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_ddt.par' ))

			# sgtf these regressors, because they are estimated from the non-sgtf data
			mcf_sgtf = mcf - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf.T]).T
			mcf_dt_sgtf = mcf_dt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_dt.T]).T
			mcf_ddt_sgtf = mcf_ddt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_ddt.T]).T

			## button presses
			self.logger.info('loading button press times')
			button_L = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_L.txt'))
			button_L_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_L]
			button_R = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_R.txt'))
			button_R_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_R]

			## blinks
			self.logger.info('loading blink times')
			this_blink_events = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'blink_times.txt'))
			blink_times_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in this_blink_events]

			## mean WM / GM / CSF volume 
			self.logger.info('loading GM / WM / CSF')
			GM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_0','resampled'])).data.astype(bool)
			WM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_1','resampled'])).data.astype(bool)
			CSF_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_2','resampled'])).data.astype(bool)

			self.logger.info('computing GM / WM / CSF mean per slice - medianed with 4 neighboring slices')
			# meaning data over voxels
			mean_GM_per_slice = np.array([np.mean(nii_data[:,sl,GM_mask[sl]],axis=1) for sl in range(n_slices)])
			mean_WM_per_slice = np.array([np.mean(nii_data[:,sl,WM_mask[sl]],axis=1) for sl in range(n_slices)])
			mean_CSF_per_slice = np.array([np.mean(nii_data[:,sl,CSF_mask[sl]],axis=1) for sl in range(n_slices)])
			# compute median over current slice +/- 3 neighboring slices
			mean_GM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_GM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			mean_WM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_WM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			mean_CSF_per_slice_sm = np.array([[sp.stats.nanmedian(mean_CSF_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
			# fill those slices with zeros that contain nans
			mean_GM_per_slice_sm_nonan = np.array([mean_GM_per_slice_sm[:,sl] if np.isnan(mean_GM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			mean_WM_per_slice_sm_nonan = np.array([mean_WM_per_slice_sm[:,sl] if np.isnan(mean_WM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			mean_CSF_per_slice_sm_nonan = np.array([mean_CSF_per_slice_sm[:,sl] if np.isnan(mean_CSF_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
			# subtract mean signal from these regressors, so that they do not start sucking op baseline activation
			mean_GM_per_slice_sm_nonan_demeaned = (mean_GM_per_slice_sm_nonan.T-np.mean(mean_GM_per_slice_sm_nonan,axis=1)).T
			mean_WM_per_slice_sm_nonan_demeaned = (mean_WM_per_slice_sm_nonan.T-np.mean(mean_WM_per_slice_sm_nonan,axis=1)).T
			mean_CSF_per_slice_sm_nonan_demeaned = (mean_CSF_per_slice_sm_nonan.T-np.mean(mean_CSF_per_slice_sm_nonan,axis=1)).T

			## retroicor regressors
			self.logger.info('loading retroicor regressors')
			retroicor_dir = os.path.join(self.runFolder('processed/mri/',run=r),'retroicor')
			retroicor_regressors = np.squeeze(np.array([NiftiImage(os.path.join(retroicor_dir,'retroicorev00%d.nii.gz'%(reg+1))).data if reg < 9 else NiftiImage(os.path.join(retroicor_dir,'retroicorev0%d.nii.gz'%(reg+1))).data for reg in np.arange(34)]))

			# n_regressors = 59#63
			# regressor_names = ['baseline','moco_x','moco_y','moco_z','moco_roll','moco_pitch','moco_yaw',
			# 		'moco_dt_x','moco_dt_y','moco_dt_z','moco_dt_roll','moco_dt_pitch','moco_dt_yaw',
			# 		'moco_ddt_x','moco_ddt_y','moco_ddt_z','moco_ddt_roll','moco_ddt_pitch','moco_ddt_yaw',
			# 		'mean_GM','mean_WM','mean_CSF',
			# 		'card1','card2','card3','card4','card5','card6',
			# 		'resp1','resp2','resp3','resp4',
			# 		'int1','int2','int3','int4','int5','int6','int7','int8',
			# 		'int9','int10','int11','int12','int13','int14','int15','int16',
			# 		'int17','int18','int19','int20','int21','int22','int23','int24',
			# 		'blinks','button_L','button_R',
			# 		'Fix_transients','Color_transients','Speed_transients','bar_stimulus'] #these regressors need not be taken into account when computing residuals
	

			import time as t
			start_time = t.time()

			slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
			non_zero_slices = [sl for sl in range(n_slices) if sl in slices]
			self.logger.info('starting nuisance GLM in parallel over slices on %d voxels'%cortex_mask.sum())

			all_residuals = []
			all_residuals = Parallel(n_jobs = n_jobs, verbose = 9)(delayed(nuisance_GLM_one_slice)
				(sl = sl,
				TR = rtime,
				n_TRs = n_timepoints,
				cortex_mask=cortex_mask,
				design_regressors =	[blink_times_list,button_L_list,button_R_list,transients_0_list,transients_1_list,transients_2_list],
				hrf_parameters = all_hrf_parameters,
				other_regressors = np.vstack([mcf_sgtf.T,mcf_dt_sgtf.T,mcf_ddt_sgtf.T,mean_GM_per_slice_sm_nonan_demeaned[sl],mean_WM_per_slice_sm_nonan_demeaned[sl],
					mean_CSF_per_slice_sm_nonan_demeaned[sl],retroicor_regressors[:,:,sl]]),
				stim_prediction = stim_predictions,
				data = nii_data,
				n_slices=n_slices,
				plot = plot,
				plotdir = this_plot_dir,
				roi_names = roi_names
				) for sl in non_zero_slices)
		
			total_elapsed_time = (t.time() - start_time)/60.
			self.logger.info( 'Finished nuisance GLM for this run in %.1f min'%(total_elapsed_time) )

			residuals = np.zeros_like(nii_data)
			residuals[:,cortex_mask] = np.array(list(chain.from_iterable(all_residuals))).T

			self.logger.info(' outputting residuals to %s'%os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res'] + res_postFix))[-1])		
			res_nii_file = NiftiImage(residuals)
			res_nii_file.header = nii_file.header
			res_nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['res'] + res_postFix))

			# delete variables to empty memory
			del(residuals)



	def hrf_from_mapper_he(self,postFix=['mcf','warped','sgtf','psc'],n_slices=30,mask='bet_mask',plot=True,target_session=-1,n_jobs=-1):
		
		self.logger.info('loading cope4 cluster-thresholded z-stat mask results')
		cortex_mask = NiftiImage(os.path.join(self.stageFolder('processed/mri/Mapper/feat_results/cope4/'),'thresh_zstat4.nii.gz')).data.astype(bool)
		# cutoff_z_score = np.sort(np.ravel(cope4))[-1001]
		# cortex_mask = (cope4>cutoff_z_score)
		# cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder( stage = 'processed/mri/masks/anat'), mask)).data, dtype = bool)

		# first, lets concatenate all the data and all the predictors
		all_data = []
		conditions = []
		onsets = []
		run_delay = 0

		for r in [self.runList[i] for i in self.conditionDict['Mapper']]: 

			TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).rtime
			if TR > 1000:
				TR /= 1000.
			if TR < 0.01:
				TR *= 1000
			n_TRs = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints
			header = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).header

			self.logger.info('loading nifti run %s'%r.ID)
			all_data.append( NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).data)

			self.logger.info('loading trial times %s'%r.ID)
			trial_times = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'trial_times.txt'))
			conditions.extend(trial_times[trial_times[:,0]!=4,0])
			onsets.extend(trial_times[trial_times[:,0]!=4,1]+run_delay)

			run_delay += n_TRs*TR

		voxels = np.vstack(all_data)[:,cortex_mask]
		del(all_data)

		drifts = np.ones((voxels.shape[0], 1))
		hrfs, betas,dm = he.glm(conditions=np.array(conditions), onsets=np.array(onsets), TR=1.6, Y=voxels, drifts=drifts, mode='r1glm', basis='3hrf', verbose=9,n_jobs=n_jobs,return_design_matrix = True)
		

		mean_hrf_params = [np.mean(hrfs[0]),np.mean(hrfs[1]),np.mean(hrfs[2])]
		all_hrfs = np.zeros([3]+list(cortex_mask.shape))
		all_hrfs[:,cortex_mask] = hrfs
		all_hrfs[:,cortex_mask == False] = np.tile(mean_hrf_params,(np.sum(cortex_mask == False),1)).T
		all_mean_hrfs = np.reshape(np.tile(mean_hrf_params,(np.size(cortex_mask),1)).T,([3] + list(cortex_mask.shape)))

		f = pl.figure(figsize=(24,24))
		xx = np.arange(0,32,0.05)

		# for mi,mask in enumerate(['V1','V2','V3','V4','LO','MT','V3AB','VO','PHC','IPS0','IPS1','IPS2','IPS3','IPS4']):#,'IPS5','FEF']
		
		# cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder( stage = 'processed/mri/masks/anat'), mask)).data, dtype = bool)

		# s = f.add_subplot(7,2,mi+1)
		s = f.add_subplot(111)
		generated_hrfs = all_hrfs[0,cortex_mask] * he.hrf.spmt(xx)[:, None] + all_hrfs[1,cortex_mask]  * he.hrf.dspmt(xx)[:, None] + all_hrfs[2,cortex_mask]  * he.hrf.ddspmt(xx)[:, None]
		mean_hrf = mean_hrf_params[0] * he.hrf.spmt(xx) +mean_hrf_params[1]* he.hrf.dspmt(xx) +mean_hrf_params[2] * he.hrf.ddspmt(xx)
		pl.plot(xx, generated_hrfs,alpha=0.05)
		pl.plot(xx, mean_hrf,'--k')
		pl.title(mask)
		pl.ylim((-.5, 1.))

		pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'),'all_HRFs'))

		self.logger.info('outputting hrf parameters')			
		all_hrf_nii_file = NiftiImage(all_hrfs)
		all_hrf_nii_file.header = header
		all_hrf_nii_file.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'hrf_parameters.nii.gz'))

		self.logger.info('outputting mean hrf parameters')			
		mean_hrf_nii_file = NiftiImage(all_mean_hrfs)
		mean_hrf_nii_file.header = header
		mean_hrf_nii_file.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'mean_hrf_parameters.nii.gz'))

		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'mean_hrf_parameters.npy'),mean_hrf_params)



		# params = Parameters()
		# params.add('hrf_a1',value=6,expr='hrf_d1/hrf_b1')
		# params.add('hrf_a2',value=12,expr='hrf_d2/hrf_b2')
		# params.add('hrf_b1',value=0.9)
		# params.add('hrf_b2',value=0.9)
		# params.add('hrf_c',value=0.35,min=0)
		# params.add('hrf_d1',value=5.4)
		# params.add('hrf_d2',value=10.8,expr='hrf_d1+delta_hrf_d')
		# params.add('delta_hrf_d',value=5.4,min=0)
			
		# regressor_list = [all_no_color_no_speed_times,all_no_color_yes_speed_times,all_yes_color_no_speed_times,all_yes_color_yes_speed_times]

		# # first, let's do a GLM with standard HRF to estimate the baseline level for every voxel
		# # the reason we don't include a baseline regressor in the GLM below (in the residual function),
		# # is that we want to exlucde the possibility that the hrf parameters might start absorbing baseline 
		# # differences (from 0). The reason that we do not want to set the betas at the level that is now fit
		# # with the canonical double gamma is that the hrf parameters might also suck up this variance. 
		# # This becomes clear when thinking about it a bit more. If an HRF rises earlier, this also means that
		# # it get's added up to itself more during convolution, resulting in a higher model prediction. 
		# run_design = NewDesign(data.shape[0], TR, sample_duration = TR/float(data.shape[1]))
		# run_design.configure(regressor_list, 
		# 	hrf_type = 'doubleGamma', 
		# 	hrf_parameters = 
		# 	{'a1' : params['hrf_a1'].value,
		# 	 'a2' : params['hrf_a2'].value, 
		# 	 'b1' : params['hrf_b1'].value, 
		# 	 'b2' : params['hrf_b2'].value, 
		# 	 'c' : params['hrf_c'].value})		

		# betas = np.zeros(([5]+list(cortex_mask.shape)))
		# baselines = np.zeros_like(data)
		# predictions= np.zeros_like(data)

		# # now loop over slices and perform glm
		# for sl in np.arange(data.shape[1]):

		# 	slice_mask = np.zeros_like(cortex_mask).astype('bool')
		# 	slice_mask[sl,:,:] = cortex_mask[sl,:,:]

		# 	if slice_mask.sum() > 0:

		# 		dm = np.mat(np.vstack([run_design.convolved_design_matrix[:,sl::n_slices],np.ones(np.shape(data)[0])]).T)
		# 		these_voxels = data[:,slice_mask]

		# 		these_betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(these_voxels)
		# 		baselines[:,slice_mask] = np.mat(dm[:,4]) * np.mat(these_betas[4,:])
		# 		betas[:,slice_mask] = these_betas
		# 		these_predictions= (np.mat(dm) * np.mat(these_betas))
		# 		predictions[:,slice_mask] = these_predictions

		# r_squareds_standard.append([])

		# now, we can subtract the baseline from the data
		# data -= baselines

		# def residual(params,data,regressor_list,TR,cortex_mask):

		# 	## create design matrix at slice time resolution
		# 	run_design = NewDesign(data.shape[0], TR, sample_duration = TR/float(data.shape[1]))
		# 	run_design.configure(regressor_list, 
		# 		hrf_type = 'doubleGamma', 
		# 		hrf_parameters = 
		# 		{'a1' : params['hrf_a1'].value,
		# 		 'a2' : params['hrf_a2'].value, 
		# 		 'b1' : params['hrf_b1'].value, 
		# 		 'b2' : params['hrf_b2'].value, 
		# 		 'c' : params['hrf_c'].value})				

		# 	residuals = np.zeros_like(data)
		# 	predictions= np.zeros_like(data)
		# 	# now loop over slices and perform glm
		# 	for sl in np.arange(data.shape[1]):

		# 		slice_mask = np.zeros_like(cortex_mask).astype('bool')
		# 		slice_mask[sl,:,:] = cortex_mask[sl,:,:]

		# 		if slice_mask.sum() > 0:

		# 			joined_design_matrix = np.mat(np.vstack([run_design.convolved_design_matrix[:,sl::n_slices]])).T
		# 			# dm[np.isnan(dm)] = 0

		# 			# sgtf all regressors, because the data is also filtered
		# 			# joined_design_matrix_sgtf = np.array([reg-sp.signal.savgol_filter(reg,window_length=int(120/TR),polyorder=3) for reg in joined_design_matrix.T]).T

		# 			# z-score all regressors, so that we no longer need a baseline parameter
		# 			joined_design_matrix_z = (joined_design_matrix-sp.mean(joined_design_matrix,axis=0)) / np.std(joined_design_matrix,axis=0)

		# 			dm = np.mat(joined_design_matrix_z)
		# 			# dm = np.mat(joined_design_matrix)
		# 			dm[np.isnan(dm)] = 0

		# 			these_voxels = data[:,slice_mask]
		# 			# z-score voxels, so that we no longer need a baseline parameter
		# 			these_voxels = (these_voxels - np.mean(these_voxels,axis=0)) / np.std(these_voxels,axis=0)
		# 			these_betas = (np.linalg.pinv((dm.T * dm)) * dm.T) * np.mat(these_voxels)
		# 			# these_betas = betas[:-1,slice_mask]
		# 			these_predictions= (np.mat(dm) * np.mat(these_betas))
		# 			predictions[:,slice_mask] = these_predictions
		# 			residuals[:,slice_mask] = these_voxels - (np.mat(dm) * np.mat(these_betas))

		# 	return np.ravel(residuals)

		# import time
		# t0 = time.time()
		# self.logger.info('now fitting hrf parameters')
		# minimize(residual, params, args=(), kws={'data':data,'regressor_list':regressor_list,'TR':TR,'cortex_mask':cortex_mask},method='powell')
		# t1 = time.time()
		# self.logger.info('done fitting in %.2f minutes'%((t1-t0)/60))
	
		# param_dict = {}
		# for param in params:
		# 	param_dict[param] = params[param].value

		# all_params.append(param_dict)


		# with open(os.path.join(self.stageFolder('processed/mri/PRF/'), 'hrf_params.pickle'), 'w') as f:
		# 	pickle.dump({'hrf_params':param_dict}, f)

		# ssr = 20
		# hrf_kernel = doubleGamma(np.arange(0,32,TR/float(ssr)),
		# 	params['hrf_a1'].value,params['hrf_a2'].value,params['hrf_b1'].value,params['hrf_b2'].value,params['hrf_c'].value)
		# hrf_kernel /= np.sum(np.abs((hrf_kernel)))
		# canonical_hrf_kernel = doubleGamma(np.arange(0,32,self.TR/float(ssr)))
		# canonical_hrf_kernel /= np.sum(np.abs((canonical_hrf_kernel)))
		# plotdir = self.stageFolder('processed/mri/figs')
		# f = pl.figure(figsize=(12,6))
		# s = f.add_subplot(111)
		# pl.plot(hrf_kernel,label='subject specific HRF')
		# pl.plot(canonical_hrf_kernel,'--k',label='canonical HRF')
		# s.legend(fancybox = True, loc = 'best')
		# pl.xticks(np.linspace(0,len(hrf_kernel),16),np.arange(0,32,2))
		# pl.xlabel('time (s)')
		# s.text(len(hrf_kernel)*0.6,0.8,'\nHRF parameters: \n\na1: %.2f\na2: %.2f\nb1: %.2f\nb2: %.2f\nc: %.2f'
		# 	 %(params['hrf_a1'],params['hrf_a2'],params['hrf_b1'],params['hrf_b2'],params['hrf_c']),horizontalalignment='center',verticalalignment='center',fontsize=12,bbox={'facecolor':'white', 'alpha':1, 'pad':10})
		# pl.savefig(os.path.join(plotdir,'HRF'))


		# # create example voxel plot difference betwee hrfs
		# f = pl.figure()

	def convert_prob_atlas_to_surface_overlay(self,label_folder = 'ProbAtlas_v4'):

		prob_label_nii_dir_input = '/home/fs_subjects/%s/label/%s/subj_vol_all/'%(self.subject.standardFSID,label_folder) 
		prob_label_nii_dir_output = '/home/fs_subjects/%s/label/%s/subj_vol_all_warped/'%(self.subject.standardFSID,label_folder) 
		prob_label_dir_output = '/home/fs_subjects/%s/label/%s/subj_vol_all_warped_labels/'%(self.subject.standardFSID,label_folder) 

		if not os.path.isdir(prob_label_nii_dir_output): os.mkdir(prob_label_nii_dir_output)
		if not os.path.isdir(prob_label_dir_output): os.mkdir(prob_label_dir_output)

		for filename in	subprocess.Popen('ls ' + prob_label_nii_dir_input + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]:
			this_roi = filename.split('/')[-1].split('.')[0]

			# lets first convert all the MNI nifti files to subject space:
			flO = FlirtOperator(inputObject = filename, referenceFileName = self.runFile(stage = 'processed/mri/', run = self.runList[self.conditionDict['PRF'][0]], postFix = ['mcf']))
			flO.configureApply(self.runFile(stage = 'processed/mri/reg/feat', base = 'standard2example_func', extension = '.mat' ), 
									outputFileName = os.path.join(prob_label_nii_dir_output, this_roi +'.nii.gz') )
			flO.execute()

			# then lets create a .mgz file of this
			vsO = VolToSurfOperator(inputObject = os.path.join(prob_label_nii_dir_output, this_roi +'.nii.gz') )
			ofn = os.path.join(self.stageFolder('processed/mri/PRF/surf/'), this_roi )
			vsO.configure(outputFileName = ofn, threshold = 0.0, surfType = 'paint' ,register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ) )
			vsO.execute()

		# remove useless files
		for filename in	subprocess.Popen('ls ' + self.stageFolder('processed/mri/PRF/surf/') + '*' + 'mgz', shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]:
			this_roi = filename.split('/')[-1]

			if (('lh' in this_roi)*('rh' in this_roi) + ('sig-0-' in this_roi)):
				# print filename
				os.remove(filename)


### code to orthonormalize dm
design =  NewDesign(n_TRs, TR, sample_duration = TR/float(n_slices))
design.configure(design_regressors, 
	hrf_parameters = hrf_params)

# if no stim:
# joined_design_matrix = other_regressors.T
# if stim:
# joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors]).T
# both cases no orth:
# joined_design_matrix_with_basleine = np.hstack([np.ones((n_TRs,1)),joined_design_matrix])
# dm = np.mat(joined_design_matrix_with_basleine)

# # if orth:
# joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors]).T
# orthonormal_dm = np.linalg.qr(joined_design_matrix)[0]
# orthonormal_dm_with_baseline = np.hstack([np.ones((n_TRs,1)),orthonormal_dm])
# dm = np.mat(orthonormal_dm_with_baseline)

# before:

joined_design_matrix = np.vstack([stim_prediction[:,slice_mask][:,voxno],other_regressors,design.convolved_design_matrix[:,sl::n_slices]]).T
orthonormal_dm = np.linalg.qr(joined_design_matrix)[0]
orthonormal_dm_with_baseline = np.hstack([np.ones((n_TRs,1)),orthonormal_dm])
dm = np.mat(orthonormal_dm_with_baseline)

these_voxels = data[:,slice_mask][:,voxno]

### from mapper glm

## mean WM / GM / CSF volume 
# self.logger.info('loading GM / WM / CSF')
# GM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_0','resampled'])).data.astype(bool)
# WM_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_1','resampled'])).data.astype(bool)
# CSF_mask = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[self.scanTypeDict['inplane_anat'][target_session]],postFix=['NB','seg_2','resampled'])).data.astype(bool)

# self.logger.info('computing GM / WM / CSF mean per slice - medianed with 4 neighboring slices')
# # meaning data over voxels
# mean_GM_per_slice = np.array([np.mean(nii_data[:,sl,GM_mask[sl]],axis=1) for sl in range(n_slices)])
# mean_WM_per_slice = np.array([np.mean(nii_data[:,sl,WM_mask[sl]],axis=1) for sl in range(n_slices)])
# mean_CSF_per_slice = np.array([np.mean(nii_data[:,sl,CSF_mask[sl]],axis=1) for sl in range(n_slices)])
# # compute median over current slice +/- 4 neighboring slices
# mean_GM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_GM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
# mean_WM_per_slice_sm = np.array([[sp.stats.nanmedian(mean_WM_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
# mean_CSF_per_slice_sm = np.array([[sp.stats.nanmedian(mean_CSF_per_slice[np.max([0,sl-4]):np.min([n_slices,sl+4]),timepoint]) for sl in range(n_slices)] for timepoint in range(n_timepoints)])
# # fill those slices with zeros that contain nans
# mean_GM_per_slice_sm_nonan = np.array([mean_GM_per_slice_sm[:,sl] if np.isnan(mean_GM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
# mean_WM_per_slice_sm_nonan = np.array([mean_WM_per_slice_sm[:,sl] if np.isnan(mean_WM_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
# mean_CSF_per_slice_sm_nonan = np.array([mean_CSF_per_slice_sm[:,sl] if np.isnan(mean_CSF_per_slice_sm[:,sl]).sum() == 0 else np.zeros(n_timepoints) for sl in range(n_slices) ])
# # subtract mean signal from these regressors, so that they do not start sucking op baseline activation
# mean_GM_per_slice_sm_nonan_demeaned = (mean_GM_per_slice_sm_nonan.T-np.mean(mean_GM_per_slice_sm_nonan,axis=1)).T
# mean_WM_per_slice_sm_nonan_demeaned = (mean_WM_per_slice_sm_nonan.T-np.mean(mean_WM_per_slice_sm_nonan,axis=1)).T
# mean_CSF_per_slice_sm_nonan_demeaned = (mean_CSF_per_slice_sm_nonan.T-np.mean(mean_CSF_per_slice_sm_nonan,axis=1)).T

