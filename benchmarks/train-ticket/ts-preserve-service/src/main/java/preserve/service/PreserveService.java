package preserve.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import preserve.entity.OrderTicketsInfo;

/**
 * @author fdse
 */
public interface PreserveService {

    Response preserve(OrderTicketsInfo oti, HttpHeaders headers);
}
