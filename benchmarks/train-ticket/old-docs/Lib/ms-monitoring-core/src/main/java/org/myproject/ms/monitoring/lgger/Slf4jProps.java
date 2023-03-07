package org.myproject.ms.monitoring.lgger;

import org.springframework.boot.context.properties.ConfigurationProperties;


@ConfigurationProperties("spring.sleuth.log.slf4j")
public class Slf4jProps {

	
	private boolean enabled = true;

	
	private String nameSkipPattern = "";

	public boolean isEnabled() {
		return this.enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	public String getNameSkipPattern() {
		return this.nameSkipPattern;
	}

	public void setNameSkipPattern(String nameSkipPattern) {
		this.nameSkipPattern = nameSkipPattern;
	}
}
