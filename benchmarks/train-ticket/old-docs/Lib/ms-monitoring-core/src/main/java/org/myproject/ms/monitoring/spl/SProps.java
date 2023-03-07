package org.myproject.ms.monitoring.spl;

import org.springframework.boot.context.properties.ConfigurationProperties;


@ConfigurationProperties("spring.sleuth.sampler")
public class SProps {

	
	private float percentage = 0.1f;

	public float getPercentage() {
		return this.percentage;
	}

	public void setPercentage(float percentage) {
		this.percentage = percentage;
	}
}
