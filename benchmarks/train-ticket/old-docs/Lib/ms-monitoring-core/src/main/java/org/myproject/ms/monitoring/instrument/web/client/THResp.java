

package org.myproject.ms.monitoring.instrument.web.client;

import java.io.IOException;
import java.io.InputStream;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.client.ClientHttpResponse;


public class THResp implements ClientHttpResponse {

	private final ClientHttpResponse delegate;
	private final TRTInter interceptor;

	public THResp(TRTInter interceptor,
			ClientHttpResponse delegate) {
		this.interceptor = interceptor;
		this.delegate = delegate;
	}

	@Override
	public HttpHeaders getHeaders() {
		return this.delegate.getHeaders();
	}

	@Override
	public InputStream getBody() throws IOException {
		return this.delegate.getBody();
	}

	@Override
	public HttpStatus getStatusCode() throws IOException {
		return this.delegate.getStatusCode();
	}

	@Override
	public int getRawStatusCode() throws IOException {
		return this.delegate.getRawStatusCode();
	}

	@Override
	public String getStatusText() throws IOException {
		return this.delegate.getStatusText();
	}

	@Override
	public void close() {
		try {
			this.delegate.close();
		}
		finally {
			this.interceptor.finish();
		}
	}
}
