
for i in range(n_points):
    		for j in range(n_points):
			d = np.linalg.norm(extractor1.cov_mats[i] - extractor2.cov_mats[j], ord='fro')
			if d < 0.3:
				colors1[i] = 'green'
				colors2[j] = 'green'

				x_lines.append(extractor1.keypoints[i][0])
				x_lines.append(extractor2.keypoints[j][0]+c)
				x_lines.append(None)

				y_lines.append(extractor1.keypoints[i][1])
				y_lines.append(extractor2.keypoints[j][1])
				y_lines.append(None)

				z_lines.append(extractor1.keypoints[i][2])
				z_lines.append(extractor2.keypoints[j][2])
				z_lines.append(None)

	fig = go.Figure()
	fig.add_trace(
		go.Scatter3d(
			x = extractor1.point_cloud[:, 0],
			y = extractor1.point_cloud[:, 1],
			z = extractor1.point_cloud[:, 2],
			mode='markers',
			marker=dict(
				size=1,
			)
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x = extractor1.keypoints[:, 0],
			y = extractor1.keypoints[:, 1],
			z = extractor1.keypoints[:, 2],
			mode='markers',
			marker=dict(
				size=3,
				color=colors1
			)
		)
	)
	extractor2.keypoints[:, 0] += c
	extractor2.point_cloud[:, 0] += c
	fig.add_trace(
		go.Scatter3d(
			x = extractor2.point_cloud[:, 0],
			y = extractor2.point_cloud[:, 1],
			z = extractor2.point_cloud[:, 2],
			mode='markers',
			marker=dict(
				size=1,
			)
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x = extractor2.keypoints[:, 0],
			y = extractor2.keypoints[:, 1],
			z = extractor2.keypoints[:, 2],
			mode='markers',
			marker=dict(
				size=3,
				color=colors2
			)
		)
	)

	fig.add_trace(
		go.Scatter3d(
			x = x_lines,
			y = y_lines,
			z = z_lines,
			mode='lines',
		)
	)
	fig.show()
	
